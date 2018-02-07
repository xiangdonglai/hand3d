#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys

from nets.E2ENet import E2ENet
from data.BinaryDbReaderSTB import BinaryDbReaderSTB
from data.BinaryDbReader import BinaryDbReader
from data.DomeReader import DomeReader
from utils.general import LearningRateScheduler
from utils.multigpu import average_gradients
from tensorflow.python.client import device_lib

def visualize(heatmap_3d, image_crop):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils.general import detect_keypoints_3d, plot_hand_3d

    heatmap_3d = np.squeeze(heatmap_3d)
    image_crop = np.squeeze(255*(image_crop+0.5)).astype(np.uint8)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(image_crop[0])
    ax = fig.add_subplot(122, projection='3d')
    keypoints = detect_keypoints_3d(heatmap_3d[0]).astype(np.float32)
    plot_hand_3d(keypoints, ax)
    plt.show()

num_gpu = sum([_.device_type == 'GPU' for _ in device_lib.list_local_devices()])
fine_tune = True
already_trained = 11000
train_para = {'lr': [1e-4, 1e-5],
              'lr_iter': [int(40000/num_gpu)],
              'max_iter': int(80000/num_gpu),
              'show_loss_freq': 100,
              'snapshot_freq': int(2000/num_gpu),
              'snapshot_dir': 'snapshots_e2e_a4-STB_heatmap',
              'loss_weight_2d': 10.0,
              'model_2d': 'snapshots_cpm_rotate_s10_wrist_vgg/model-60000.pickle'
              }

PATH_TO_SNAPSHOTS = './{}/model-{}'.format(train_para['snapshot_dir'], already_trained)  # only used when USE_RETRAINED is true
lifting_dict = {'method': 'heatmap'}

with tf.Graph().as_default(), tf.device('/cpu:0'):
    # get dataset
    # dataset = DomeReader(mode='training', flip_2d=True, applyDistort=True,
    #                          batch_size=8, shuffle=True, hand_crop=True, use_wrist_coord=True, crop_size=368, sigma=10.0,
    #                          crop_size_zoom=2.0, crop_center_noise=True, crop_offset_noise=True, crop_scale_noise=True, a4=True, a2=False)
    dataset = BinaryDbReaderSTB(mode='training', batch_size=8, shuffle=True, hand_crop=True, use_wrist_coord=True, crop_size=368, sigma=10.0, crop_size_zoom=2.0, crop_center_noise=True, 
        crop_offset_noise=True, crop_scale_noise=True)
    # dataset = BinaryDbReader(mode='training', batch_size=8, shuffle=True, hand_crop=True, use_wrist_coord=False, crop_size=368, sigma=10.0, crop_size_zoom=2.0, crop_center_noise=True, 
    #    crop_offset_noise=True, crop_scale_noise=True)

    # build network graph
    data = dataset.get(extra=True)

    for k, v in data.items():
        data[k] = tf.split(v, num_gpu, 0)

    # Solver
    if fine_tune:
        global_step = tf.Variable(already_trained+1, trainable=False, name="global_step")
    else:
        global_step = tf.Variable(0, trainable=False, name="global_step")
    lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
    lr = lr_scheduler.get_lr(global_step)
    opt = tf.train.AdamOptimizer(lr)

    tower_grads  = []
    tower_losses = []
    tower_losses_3d = []
    tower_losses_2d = []

    tower_losses_detail = {'pose': [], 'view': []}

    evaluation = tf.placeholder_with_default(False, shape=())

    with tf.variable_scope(tf.get_variable_scope()):
        for ig in range(num_gpu):
            with tf.device('/gpu:%d' % ig):

                # build network
                net = E2ENet(lifting_dict, out_chan=22, crop_size=368)
                rel_dict = net.inference(data['image_crop'][ig], evaluation, train=True)

                # Loss
                predicted_scoremaps = rel_dict['heatmap_2d']
                assert len(predicted_scoremaps) == 6

                s = data['scoremap'][ig].get_shape().as_list()
                ext_vis = tf.concat([data['keypoint_vis21'][ig], tf.ones([s[0], 1], dtype=tf.bool)], axis=1)
                vis = tf.cast(tf.reshape(ext_vis, [s[0], s[3]]), tf.float32)
                loss_2d = 0.0
                for ip, predicted_scoremap in enumerate(predicted_scoremaps):
                    resized_scoremap = tf.image.resize_images(predicted_scoremap, (s[1], s[2]))
                    loss_2d += tf.reduce_sum(vis * tf.reduce_mean(tf.square(resized_scoremap - data['scoremap'][ig]), [1, 2])) / (tf.reduce_sum(vis) + 0.001)
                    loss_2d /= len(predicted_scoremaps)

                if lifting_dict['method'] == 'direct':
                    loss_pose = tf.reduce_mean(tf.square(rel_dict['coord_xyz_can'] - data['keypoint_xyz21_can'][ig]))
                    loss_view = tf.reduce_mean(tf.square(rel_dict['rot_mat'] - data['rot_mat'][ig]))
                    loss_3d = loss_pose + loss_view
                    tower_losses_detail['pose'].append(loss_pose)
                    tower_losses_detail['view'].append(loss_view)
                elif lifting_dict['method'] == 'heatmap':
                    predicted_scoremaps_3d = rel_dict['heatmap_3d']
                    loss_3d = 0.0
                    for ip, predicted_scoremap in enumerate(predicted_scoremaps_3d): 
                        loss_3d += tf.reduce_sum(vis * tf.reduce_mean(tf.square(predicted_scoremap - data['scoremap_3d'][ig]), [1, 2, 3])) / (tf.reduce_sum(vis) + 0.001)
                        loss_3d /= len(predicted_scoremaps_3d)

                loss = loss_3d + loss_2d * train_para['loss_weight_2d']
                tf.get_variable_scope().reuse_variables()

                tower_losses.append(loss)
                tower_losses_3d.append(loss_3d)
                tower_losses_2d.append(loss_2d)
                grad = opt.compute_gradients(loss)
                tower_grads.append(grad)

    total_loss = tf.reduce_mean(tower_losses)
    total_loss_3d = tf.reduce_mean(tower_losses_3d)
    total_loss_2d = tf.reduce_mean(tower_losses_2d)
    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    total_loss_detail = {k: tf.reduce_mean(v) for k, v in tower_losses_detail.items() if len(v) > 0}

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_3d', loss_3d)
    tf.summary.scalar('loss_2d', loss_2d)

    # init weights
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(train_para['snapshot_dir'] + '/train',
                                          sess.graph)
    if not fine_tune:
        start_iter = 0
        net.init(sess, weight_files=['snapshots_e2e_heatmap/model-75000.pickle'])
        # net.init(sess, weight_files=[train_para['model_2d']])
        # net.init(sess, cpm_init_vgg=True)
    else:
        saver.restore(sess, PATH_TO_SNAPSHOTS)
        start_iter = already_trained + 1

    # snapshot dir
    if not os.path.exists(train_para['snapshot_dir']):
        os.mkdir(train_para['snapshot_dir'])
        print('Created snapshot dir:', train_para['snapshot_dir'])

    # Training loop
    print('Starting to train ...')
    for i in range(start_iter, train_para['max_iter']):
        if lifting_dict['method'] == 'direct':
            summary, _, loss_v, loss_3d_v, loss_2d_v, loss_pose_v, loss_view_v = sess.run([merged, apply_gradient_op, loss, loss_3d, loss_2d, total_loss_detail['pose'], total_loss_detail['view']])
        elif lifting_dict['method'] == 'heatmap':
            summary, _, loss_v, loss_3d_v, loss_2d_v = sess.run([merged, apply_gradient_op, loss, loss_3d, loss_2d])
        train_writer.add_summary(summary, i)

        # visualize(heatmap_3d_v, image_crop_v)

        if (i % train_para['show_loss_freq']) == 0:
            print('Iteration %d\t Loss %.1e, Loss_3d %.1e, Loss_2d %.1e' % (i, loss_v, loss_3d_v, loss_2d_v))
            if lifting_dict['method'] == 'direct':
                print('Pose: %.1e, View: %.1e' % (loss_pose_v, loss_view_v))
            sys.stdout.flush()

        if (i % train_para['snapshot_freq']) == 0:
            saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
            print('Saved a snapshot.')
            sys.stdout.flush()

    print('Training finished. Saving final snapshot.')
    saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
