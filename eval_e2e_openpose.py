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
import argparse, cv2, os
import pdb

from data.OpenposeReader import OpenposeReader
from nets.E2ENet import E2ENet
from utils.general import EvalUtil, get_stb_ref_curves, calc_auc, plot_hand_3d, detect_keypoints, trafo_coords, plot_hand, detect_keypoints_3d, hand_size, load_weights_from_snapshot

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', '-v', action='store_true')
parser.add_argument('--save', '-s', action='store_true')
parser.add_argument('--output_dir', '-o', type=str, default='/home/donglaix/Documents/Experiments/output_3d_heatmap1')
args = parser.parse_args()

# get dataset
dataset = OpenposeReader(mode='evaluation', shuffle=False, use_wrist_coord=True, hand_crop=True, crop_size=368, crop_size_zoom=2.0, flip_2d=True)

# build network graph
data = dataset.get()
image_crop = data['image_crop']
lifting_dict = {'method': 'heatmap'}

# build network
net = E2ENet(lifting_dict, out_chan=22, crop_size=368)

# feed through network
evaluation = tf.placeholder_with_default(True, shape=())
rel_dict = net.inference(image_crop, evaluation, train=False)
s = image_crop.get_shape().as_list()
heatmap_2d = tf.image.resize_images(rel_dict['heatmap_2d'][-1], (s[1], s[2]))

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

# cpt = 'snapshots_e2e/model-100000'
# cpt = 'snapshots_e2e_RHD_STB_nw/model-38000'
cpt = 'snapshots_e2e_heatmap/model-100000'
load_weights_from_snapshot(sess, cpt, discard_list=['Adam', 'global_step', 'beta', 'scale'])

util = EvalUtil()
# iterate dataset
for i in range(dataset.num_samples):
    # get prediction
    if lifting_dict['method'] == 'direct':
        keypoint_vis21, keypoint_uv21, image_crop_v, coord_xyz_norm, scoremap_v, img_dir = \
            sess.run([data['keypoint_vis21'], data['keypoint_uv21'], data['image_crop'], rel_dict['coord_xyz_norm'], heatmap_2d, data['img_dir']])
    elif lifting_dict['method'] == 'heatmap':
        keypoint_vis21, keypoint_uv21, image_crop_v, scoremap_v, scoremap_3d_v, img_dir = \
            sess.run([data['keypoint_vis21'], data['keypoint_uv21'], data['image_crop'], heatmap_2d, rel_dict['heatmap_3d'][-1], data['img_dir']])

    keypoint_vis21 = np.squeeze(keypoint_vis21)
    keypoint_uv21 = np.squeeze(keypoint_uv21)
    image_crop_v = np.squeeze((image_crop_v+0.5)*255).astype(np.uint8)
    scoremap_v = np.squeeze(scoremap_v)

    coord2d_v = detect_keypoints(scoremap_v)
    coord2d_v = coord2d_v[:21, :]

    if lifting_dict['method'] == 'direct':
        coord3d_pred_v = np.squeeze(coord_xyz_norm)
    elif lifting_dict['method'] == 'heatmap':
        scoremap_3d_v = np.squeeze(scoremap_3d_v)
        coord3d_pred_v = detect_keypoints_3d(scoremap_3d_v)
        coord3d_pred_v = coord3d_pred_v[:21, :]

    coord3d_pred_v -= coord3d_pred_v[0, :]
    # rescale to meters
    coord3d_pred_v /= hand_size(coord3d_pred_v)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

        if args.visualize:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121, projection='3d')
            plot_hand_3d(coord3d_pred_v, ax1, color_fixed=np.array([0.0, 0.0, 1.0]))
            ax1.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            plt.xlabel('x')
            plt.ylabel('y')
            ax1.set_xlim(-0.1, 0.1)
            ax1.set_ylim(-0.1, 0.1)
            ax1.set_zlim(-0.1, 0.1)

            ax2 = fig.add_subplot(122)
            plt.imshow(image_crop_v)
            plot_hand(coord2d_v, ax2, color_fixed=np.array([0.0, 0.0, 1.0]))
            plot_hand(keypoint_uv21[:, ::-1], ax2, color_fixed=np.array([1.0, 0.0, 0.0]))

            plt.show()
            # pdb.set_trace()

    if args.save:
        coord3d_pred_v[:, 0] = -coord3d_pred_v[:, 0]

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        plot_hand_3d(coord3d_pred_v, ax1, color_fixed=np.array([0.0, 0.0, 1.0]))
        plt.xlabel('x')
        plt.ylabel('y')
        ax1.view_init(azim=-90.0, elev=-70.0)
        ax1.set_xlim(-0.2, 0.2)
        ax1.set_ylim(-0.2, 0.2)
        ax1.set_zlim(-0.2, 0.2)
        fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
        
        # plt.show()

        fig.canvas.draw()
        figure = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        figure = figure.reshape(fig.canvas.get_width_height()[::-1] + (3,))[:, :, ::-1]

        img_dir = img_dir[0].decode().replace('individ_imgs', 'detected_hand1')
        img = cv2.imread(img_dir)
        dh = figure.shape[0]
        dw = int(float(img.shape[1]) / img.shape[0] * dh)
        resized_img = cv2.resize(img, (dw, dh))

        concat = np.concatenate((resized_img, figure), axis=1)
        plt.cla()
        plt.clf()
        plt.close()

        basename = os.path.basename(img_dir)
        output_file = os.path.join(args.output_dir, basename)
        cv2.imwrite(output_file, concat)