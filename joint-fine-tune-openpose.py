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

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from data.BinaryDbReaderSTB import BinaryDbReaderSTB
from openpose.OpenposeSTBReader import OpenposeSTBReader
from utils.general import LearningRateScheduler, hand_size
# import pdb

# training parameters
train_para = {'lr': [1e-5, 1e-6],
              'lr_iter': [40000],
              'max_iter': 80000,
              'show_loss_freq': 100,
              'snapshot_freq': 5000,
              'org_weight': './weights/posenet3d-dome-hs.pickle',
              'ft_snapshot_dir': 'snapshots_joint_dome_hs_jft_openpose'}
# train_para = {'lr': [1e-5, 1e-6],
#               'lr_iter': [40000],
#               'max_iter': 80000,
#               'show_loss_freq': 100,
#               'snapshot_freq': 5000,
#               'org_weight': './weights/posenet3d-my.pickle',
#               'ft_snapshot_dir': 'snapshots_joint_my_jft_openpose',
#               }

# get dataset
dataset = BinaryDbReaderSTB(mode='training',
                         batch_size=8, shuffle=False, hand_crop=False, use_wrist_coord=False)
openpose_dataset = OpenposeSTBReader(mode='training', batch_size=8, shuffle=False, hand_crop=True, use_wrist_coord=False)

# build network graph
data = dataset.get()
openpose_data = openpose_dataset.get()

# build network
net = ColorHandPose3DNetwork()

# feed trough network
scoremap_pooled = tf.nn.avg_pool(openpose_data['scoremap'], ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
evaluation = tf.placeholder_with_default(True, shape=())
_, coord3d_pred, R = net._inference_pose3d(scoremap_pooled, openpose_data['hand_side'], evaluation, train=True)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

# Loss
loss = 0.0
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
net.init(sess, weight_files=['./weights/handsegnet-rhd.pickle', train_para['org_weight']], exclude_var_list=['HandSegNet', 'PoseNet2D'])
saver = tf.train.Saver(max_to_keep=None)

# snapshot dir
if not os.path.exists(train_para['ft_snapshot_dir']):
    os.mkdir(train_para['ft_snapshot_dir'])
    print('Created snapshot dir:', train_para['ft_snapshot_dir'])

# Training loop
print('Starting to train ...')
for i in range(train_para['max_iter']):
    _, loss_v, openpose_key, key = sess.run([train_op, loss, openpose_data['key'], data['key']])

    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e' % (i, loss_v))
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['ft_snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()

    assert (key == openpose_key).all()

print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['ft_snapshot_dir'], global_step=train_para['max_iter'])
