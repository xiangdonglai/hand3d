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
""" Script for evaluation of the end-to-end method.
"""""

from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse, cv2
import pdb

from data.BinaryDbReader import *
from data.BinaryDbReaderSTB import *
from nets.CPM import CPM
from utils.general import EvalUtil, get_stb_ref_curves, calc_auc, plot_hand_3d, detect_keypoints, trafo_coords, plot_hand, detect_keypoints_3d, hand_size, load_weights_from_snapshot

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', '-v', action='store_true')
parser.add_argument('--save', '-s', action='store_true')
args = parser.parse_args()

# get dataset
# dataset = BinaryDbReader(mode='evaluation', shuffle=False, use_wrist_coord=False)
dataset = BinaryDbReaderSTB(mode='evaluation', shuffle=False, use_wrist_coord=False, hand_crop=True)

# build network graph
data = dataset.get()
image_crop = (data['image_crop'])
image_crop = image_crop[:, :, ::-1, ::-1] # convert to BGR
# build network
net = CPM(out_chan=22)

# feed through network
scoremap, _ = net.inference(image_crop)[-1]

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

weight_path = './weights/pose_model.npy'
net.init(weight_path, sess)

util = EvalUtil()
# iterate dataset
for i in range(dataset.num_samples):
    # get prediction
    keypoint_xyz21, keypoint_vis21, keypoint_scale, keypoint_uv21_v, image_crop_v, scoremap_v = \
        sess.run([data['keypoint_xyz21'], data['keypoint_vis21'], data['keypoint_scale'], data['keypoint_uv21'], image_crop, scoremap])

    keypoint_xyz21 = np.squeeze(keypoint_xyz21)
    keypoint_vis21 = np.squeeze(keypoint_vis21)
    keypoint_scale = np.squeeze(keypoint_scale)
    keypoint_uv21_v = np.squeeze(keypoint_uv21_v)
    image_crop_v = np.squeeze((image_crop_v+0.5)*256).astype(np.uint8)
    scoremap_v = np.squeeze(scoremap_v)
    for ik in (1, 5, 9, 13, 17):
        scoremap_v[:, :, ik:ik+4] = scoremap_v[:, :, ik+3:ik-1:-1]

    coord2d_v = detect_keypoints(scoremap_v) * 8

    # center gt
    keypoint_xyz21 -= keypoint_xyz21[0, :]

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

        if args.visualize:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121, projection='3d')
            plot_hand_3d(keypoint_xyz21, ax1, color_fixed=np.array([1.0, 0.0, 0.0]))
            ax1.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            plt.xlabel('x')
            plt.ylabel('y')
            ax1.set_xlim(-0.1, 0.1)
            ax1.set_ylim(-0.1, 0.1)
            ax1.set_zlim(-0.1, 0.1)

            ax2 = fig.add_subplot(122)
            plt.imshow(image_crop_v)
            plot_hand(coord2d_v, ax2)

            plt.show()
            # pdb.set_trace()

