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

from nets.PosePriorNetwork import PosePriorNetwork
from data.BinaryDbReader import BinaryDbReader
from data.DomeReader import DomeReader
from utils.general import LearningRateScheduler

def visualize(scoremap, hand_side, rot_mat, coord3d_can, coord3d, coord2d):
    import pdb
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from utils.general import plot_hand_3d, plot_hand
    import numpy as np
    l = scoremap.shape[0]
    for i in range(l):
        l_scoremap = scoremap[i, :, :, :]
        l_coord3d = coord3d[i, :, :]
        l_coord3d_can = coord3d_can[i, :, :] * 6.0
        l_rot_mat = rot_mat[i, :, :]
        l_coord2d = coord2d[i, :, :]
        if hand_side[i, 1] == 1:
            print('flip')
            # l_coord3d_can[:, 2] = -l_coord3d_can[:, 2]
        # l_coord3d_rotate = np.dot(l_coord3d_can, l_rot_mat)
        l_coord3d -= l_coord3d[0, :]
        # l_coord3d_rotate *= 2.0
        s = l_scoremap.shape
        keypoint_coords = np.zeros((s[2], 2))
        for i in range(s[2]):
            v, u = np.unravel_index(np.argmax(l_scoremap[:, :, i]), (s[0], s[1]))
            keypoint_coords[i, 0] = v
            keypoint_coords[i, 1] = u
        fig = plt.figure(1)
        ax1 = fig.add_subplot(131)
        ax1.imshow(np.amax(l_scoremap, axis=2))
        ax2 = fig.add_subplot(132, projection='3d')
        plot_hand_3d(l_coord3d, ax2, color_fixed=np.array([1.0, 0.0, 1.0]))
        plot_hand_3d(l_coord3d_can, ax2, color_fixed=np.array([0.0, 1.0, 0.0]))
        ax2.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        ax3 = fig.add_subplot(133)
        plot_hand(keypoint_coords, ax3, color_fixed=np.array([0.0, 1.0, 0.0]))
        plt.gca().invert_yaxis()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

# Chose which variant to evaluate
# VARIANT = 'direct'
# VARIANT = 'bottleneck'
# VARIANT = 'local'
# VARIANT = 'local_w_xyz_loss'
VARIANT = 'proposed'

# training parameters
# train_para = {'lr': [1e-5, 1e-6],
#               'lr_iter': [60000],
#               'max_iter': 80000,
#               'show_loss_freq': 1000,
#               'snapshot_freq': 5000,
#               'snapshot_dir': 'snapshots_lifting_%s_dome' % VARIANT}
train_para = {'lr': [1e-4, 1e-5],
              'lr_iter': [60000],
              'max_iter': 120000,
              'show_loss_freq': 100,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots_lifting_%s_dome' % VARIANT}

# get dataset
# dataset = BinaryDbReader(mode='training',
#                          batch_size=8, shuffle=True, hand_crop=True, use_wrist_coord=False,
#                          coord_uv_noise=False, crop_center_noise=False, crop_offset_noise=False, crop_scale_noise=False)
# dataset = DomeReader(mode='training',
#                          batch_size=8, shuffle=True, hand_crop=True, use_wrist_coord=False,
#                          coord_uv_noise=False, crop_center_noise=False, crop_offset_noise=False, crop_scale_noise=False)
dataset = DomeReader(mode='training',
                         batch_size=8, shuffle=True, hand_crop=True, use_wrist_coord=False,
                         coord_uv_noise=True, crop_center_noise=True, crop_offset_noise=True, crop_scale_noise=True)

# build network graph
data = dataset.get()

# build network
net = PosePriorNetwork(VARIANT)

# feed trough network
evaluation = tf.placeholder_with_default(True, shape=())
_, coord3d_pred, R = net.inference(data['scoremap'], data['hand_side'], evaluation)

# cond_right = tf.equal(tf.argmax(data['hand_side'], 1), 1)
# cond_right_all = tf.tile(tf.reshape(cond_right, [-1, 1, 1]), [1, 21, 3])
# coord_xyz_can_flip = PosePriorNetwork._flip_right_hand(data['keypoint_xyz21_can'], cond_right_all)
# coord_xyz_rel_normed = tf.matmul(coord_xyz_can_flip, data['rot_mat'])

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# Loss
loss = 0.0
if (VARIANT == 'direct') or (VARIANT == 'bottleneck'):
    loss = tf.reduce_mean(tf.square(coord3d_pred - data['keypoint_xyz21_normed']))
elif VARIANT == 'local':
    loss += tf.reduce_mean(tf.square(coord3d_pred - data['keypoint_xyz21_local']))
elif VARIANT == 'local_w_xyz_loss':
    from utils.relative_trafo import bone_rel_trafo_inv

    # transform the local coordinates back into xyz for the loss
    coord3d_pred_xyz = bone_rel_trafo_inv(coord3d_pred)
    loss += tf.reduce_mean(tf.square(coord3d_pred_xyz - data['keypoint_xyz21_normed']))
elif VARIANT == 'proposed':
    loss += tf.reduce_mean(tf.square(coord3d_pred - data['keypoint_xyz21_can']))
    loss += tf.reduce_mean(tf.square(R - data['rot_mat']))

# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

# init weights
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=4.0)

# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
print('Starting to train ...')
for i in range(train_para['max_iter']):
    # _, loss_v, scoremap, hand_side, rot_mat, coord3d_can, coord3d, coord2d = sess.run([train_op, loss, data['scoremap'], data['hand_side'], data['rot_mat'], coord_xyz_rel_normed, data['keypoint_xyz21'], data['keypoint_uv21']])
    _, loss_v = sess.run([train_op, loss])

    # visualize(scoremap, hand_side, rot_mat, coord3d_can, coord3d, coord2d)

    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e' % (i, loss_v))
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()

print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
