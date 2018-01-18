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
from utils.general import LearningRateScheduler

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

# training parameters
# train_para = {'lr': [1e-5, 1e-6],
#               'lr_iter': [60000],
#               'max_iter': 80000,
#               'show_loss_freq': 1000,
#               'snapshot_freq': 5000,
#               'snapshot_dir': 'snapshots_lifting_%s_dome' % VARIANT}
train_para = {'lr': [1e-4, 1e-5],
              'lr_iter': [20000],
              'max_iter': 40000,
              'show_loss_freq': 100,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots_e2e',
              }

# get dataset
dataset = BinaryDbReaderSTB(mode='training',
                         batch_size=8, shuffle=True, hand_crop=True, use_wrist_coord=False,
                         coord_uv_noise=True, crop_center_noise=True, crop_offset_noise=True, crop_scale_noise=True)

# build network graph
data = dataset.get()

# build network
net = E2ENet(32)

# feed trough network
heatmap_3d, heatmap_2d = net.inference(data['image_crop'], train=True)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

# Loss
assert len(heatmap_2d) == 3
loss_2d = 0.0
for i, pred_item in enumerate(heatmap_2d):
    loss_2d += tf.reduce_mean(tf.square(pred_item - tf.image.resize_images(data['scoremap'], (32, 32)))) / 3
loss_3d = tf.reduce_mean(tf.square(heatmap_3d - data['scoremap_3d']))
loss = loss_3d + loss_2d
tf.summary.scalar('loss', loss)

# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

# init weights
sess.run(tf.global_variables_initializer())
# net.init(sess, weight_files=['./weights/handsegnet-rhd.pickle', train_para['org_weight']])
saver = tf.train.Saver(max_to_keep=None)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(train_para['snapshot_dir'] + '/train',
                                      sess.graph)

# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
print('Starting to train ...')
for i in range(train_para['max_iter']):
    summary, _, loss_v, heatmap_3d_v, image_crop_v, loss_3d_v, loss_2d_v = sess.run([merged, train_op, loss, data['scoremap_3d'], data['image_crop'], loss_3d, loss_2d])
    train_writer.add_summary(summary, i)

    # visualize(heatmap_3d_v, image_crop_v)

    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e, Loss_3d %.1e, Loss_2d %.1e' % (i, loss_v, loss_3d_v, loss_2d_v))
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()

print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])