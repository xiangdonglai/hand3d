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
from data.DomeReader import DomeReader
from utils.general import LearningRateScheduler, load_weights_from_snapshot

# training parameters
train_para = {'lr': [1e-4, 1e-5, 1e-6],
              'lr_iter': [10000, 20000],
              'max_iter': 30000,
              'show_loss_freq': 100,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots_cpm'}

# get dataset
dataset = DomeReader(mode='training',
                         batch_size=8, shuffle=True, use_wrist_coord=False,
                         hand_crop=True, coord_uv_noise=True, crop_center_noise=True, a4=True, a2=False)

# build network graph
data = dataset.get(read_image=True)

# build network
evaluation = tf.placeholder_with_default(True, shape=())
net = CPM()
predicted_scoremaps = net.inference(data['image_crop'], train=True)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

# Loss
loss = 0.0
for predicted_scoremap in predicted_scoremaps:
    resize = predicted_scoremap.get_shape().as_list()
    resized_scoremap = tf.image.resize_images(data['scoremap'], (resize[1], resize[2]))
    loss += tf.reduce_mean(tf.square(predicted_scoremap - resized_scoremap))
loss /= len(predicted_scoremaps)
tf.summary.scalar('loss', loss)

# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

# init weights
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(train_para['snapshot_dir'] + '/train',
                                      sess.graph)

# Training loop
print('Starting to train ...')
for i in range(train_para['max_iter']):
    summary, _, loss_v = sess.run([merged, train_op, loss])
    train_writer.add_summary(summary, i)

    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e' % (i, loss_v))
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()


print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
