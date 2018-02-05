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

from nets.CPM import CPM
from data.TsimonDBReader import TsimonDBReader
from data.DomeReader import DomeReader
from data.MultiDataset import MultiDataset
from utils.general import LearningRateScheduler, load_weights_from_snapshot
from utils.multigpu import average_gradients
from tensorflow.python.client import device_lib

num_gpu = sum([_.device_type == 'GPU' for _ in device_lib.list_local_devices()])
fine_tune = False
already_trained = 30000
PATH_TO_SNAPSHOTS = './snapshots_cpm_rotate_s10_wrist_vgg/model-{}'.format(already_trained)  # only used when USE_RETRAINED is true
# training parameters
train_para = {'lr': [1e-4, 1e-5, 1e-6],
              'lr_iter': [int(160000/num_gpu), int(20000/num_gpu)],
              'max_iter': int(200000/num_gpu),
              'show_loss_freq': 100,
              'snapshot_freq': int(5000/num_gpu),
              'snapshot_dir': 'snapshots_cpm_rotate_s10_wrist_dome_simon'}

with tf.Graph().as_default(), tf.device('/cpu:0'):
    # get dataset
    datasets = [
        TsimonDBReader(mode='training',
                             batch_size=4*num_gpu, shuffle=True, use_wrist_coord=True, crop_size=368, sigma=10.0, random_rotate=True, random_hue=False, crop_size_zoom=2.0,
                             hand_crop=True, crop_center_noise=True, crop_scale_noise=True, crop_offset_noise=True)
        # DomeReader(mode='training', flip_2d=True, applyDistort=True,
        #                      batch_size=4*num_gpu, shuffle=True, use_wrist_coord=True, crop_size=368, sigma=10.0, crop_size_zoom=2.0,
        #                      hand_crop=True, crop_center_noise=True, crop_scale_noise=True, crop_offset_noise=True, a4=True, a2=True)
    ]
    dataset = MultiDataset(datasets)

    # build network graph
    data = dataset.get(read_image=True, extra=True)
    for k, v in data.items():
        data[k] = tf.split(v, num_gpu, 0)

    tower_grads  = []
    tower_losses = []

    # Solver
    if fine_tune:
        global_step = tf.Variable(already_trained+1, trainable=False, name="global_step")
    else:
        global_step = tf.Variable(0, trainable=False, name="global_step")
    lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
    lr = lr_scheduler.get_lr(global_step)
    opt = tf.train.AdamOptimizer(lr)

    with tf.variable_scope(tf.get_variable_scope()):
        for ig in range(num_gpu):
            with tf.device('/gpu:%d' % ig):

                # build network
                net = CPM(crop_size=368, out_chan=22)
                predicted_scoremaps, _ = net.inference(data['image_crop'][ig], train=True)

                # Loss
                s = data['scoremap'][ig].get_shape().as_list()
                ext_vis = tf.concat([data['keypoint_vis21'][ig], tf.ones([s[0], 1], dtype=tf.bool)], axis=1)
                vis = tf.cast(tf.reshape(ext_vis, [s[0], s[3]]), tf.float32)
                losses = []
                loss = 0.0
                for ip, predicted_scoremap in enumerate(predicted_scoremaps):
                    resized_scoremap = tf.image.resize_images(predicted_scoremap, (s[1], s[2]))
                    losses.append(tf.reduce_sum(vis * tf.reduce_mean(tf.square(resized_scoremap - data['scoremap'][ig]), [1, 2])) / (tf.reduce_sum(vis) + 0.001))
                    loss += losses[ip]
                loss /= len(predicted_scoremaps)
                tf.get_variable_scope().reuse_variables()

                tower_losses.append(loss)
                grad = opt.compute_gradients(loss)
                tower_grads.append(grad)

    total_loss = tf.reduce_mean(tower_losses)
    grads = average_gradients(tower_grads)
    tf.summary.scalar('loss', total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    # init weights
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)

    # snapshot dir
    if not os.path.exists(train_para['snapshot_dir']):
        os.mkdir(train_para['snapshot_dir'])
        print('Created snapshot dir:', train_para['snapshot_dir'])

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(train_para['snapshot_dir'] + '/train',
                                          sess.graph)

    if not fine_tune:
        start_iter = 0
        net.init_pickle(sess, ['./snapshots_cpm_rotate_s10_wrist_dome/model-100000.pickle'])
        # net.init_vgg(sess)
    else:
        saver.restore(sess, PATH_TO_SNAPSHOTS)
        start_iter = already_trained + 1

    # Training loop
    print('Starting to train ...')
    for i in range(start_iter, train_para['max_iter']):
        summary, _, loss_v = sess.run([merged, apply_gradient_op, loss])
        train_writer.add_summary(summary, i)

        if (i % train_para['show_loss_freq']) == 0:
            print('Iteration %d\t Loss %.2e' % (i, loss_v))
            sys.stdout.flush()

        if (i % train_para['snapshot_freq']) == 0:
            saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
            print('Saved a snapshot.')
            sys.stdout.flush()


    print('Training finished. Saving final snapshot.')
    saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
