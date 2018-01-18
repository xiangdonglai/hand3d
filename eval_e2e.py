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
from nets.E2ENet import E2ENet
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
image_crop = data['image_crop']
# build network
net = E2ENet(32)

# feed through network
scoremap_3d, scoremap = net.inference(image_crop)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

cpt = 'snapshots_e2e/model-40000'
load_weights_from_snapshot(sess, cpt, discard_list=['Adam', 'global_step', 'beta'])

util = EvalUtil()
# iterate dataset
for i in range(dataset.num_samples):
    # get prediction
    keypoint_xyz21, keypoint_vis21, keypoint_scale, keypoint_uv21_v, image_crop_v, scoremap_3d_v, scoremap_v = \
        sess.run([data['keypoint_xyz21'], data['keypoint_vis21'], data['keypoint_scale'], data['keypoint_uv21'], image_crop, scoremap_3d, scoremap])

    keypoint_xyz21 = np.squeeze(keypoint_xyz21)
    keypoint_vis21 = np.squeeze(keypoint_vis21)
    keypoint_scale = np.squeeze(keypoint_scale)
    keypoint_uv21_v = np.squeeze(keypoint_uv21_v)
    image_crop_v = np.squeeze((image_crop_v+0.5)*255).astype(np.uint8)
    scoremap_3d_v = np.squeeze(scoremap_3d_v)
    scoremap_v = np.squeeze(scoremap_v)

    # rescale to meters
    coord3d_pred_v = detect_keypoints_3d(scoremap_3d_v)
    coord3d_pred_v -= coord3d_pred_v[0, :]
    coord3d_pred_v *= keypoint_scale / hand_size(coord3d_pred_v)
    coord2d_v = detect_keypoints(scoremap_v) * 8

    # center gt
    keypoint_xyz21 -= keypoint_xyz21[0, :]

    util.feed(keypoint_xyz21, keypoint_vis21, coord3d_pred_v)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

        if args.visualize:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121, projection='3d')
            plot_hand_3d(coord3d_pred_v, ax1, color_fixed=np.array([0.0, 0.0, 1.0]))
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

    if args.save:
        fig = plt.figure(figsize=(12, 6))
        keypoints2d = detect_keypoints(keypoints_scoremap_v)
        coord_hw = trafo_coords(keypoints2d, center_v, scale_v, 256)
        coord_uv21 = keypoint_uv21_v[:, ::-1]/2
        ax1 = fig.add_subplot(121)
        plt.imshow(image_scaled_v)
        plot_hand(coord_hw, ax1, color_fixed=np.array((0., 0., 1.0)))
        plot_hand(coord_uv21, ax1, color_fixed=np.array((1., 0., 0.0)))

        ax2 = fig.add_subplot(122, projection='3d')
        plot_hand_3d(coord3d_pred_v, ax2, color_fixed=np.array([0.0, 0.0, 1.0]))
        plot_hand_3d(keypoint_xyz21, ax2, color_fixed=np.array([1.0, 0.0, 0.0]))
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_xlim(-0.1, 0.1)
        ax2.set_ylim(-0.1, 0.1)
        ax2.set_zlim(-0.1, 0.1)
        ax2.view_init(azim=-90.0, elev=-65.0)  # aligns the 3d coord with the camera view
        plt.tight_layout()
        # plt.show()

        fig.canvas.draw()
        figure = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        figure = figure.reshape(fig.canvas.get_width_height()[::-1] + (3,))[:, :, ::-1]

        plt.clf()
        plt.close()

        # plt.imshow(figure)
        # plt.show()
        cv2.imwrite('../results/{0:04d}.jpg'.format(i), figure)

# Output results
mean, median, auc, pck_curve_all, threshs = util.get_measures(0.0, 0.050, 20)  # rainier: Should lead to 0.764 / 9.405 / 12.210
print('Evaluation results')
print('Average mean EPE: %.3f mm' % (mean*1000))
print('Average median EPE: %.3f mm' % (median*1000))
print('Area under curve between 0mm - 50mm: %.3f' % auc)

# only use subset that lies in 20mm .. 50mm
pck_curve_all, threshs = pck_curve_all[8:], threshs[8:]*1000.0
auc_subset = calc_auc(threshs, pck_curve_all)
print('Area under curve between 20mm - 50mm: %.3f' % auc_subset)

# Show Figure 9 from the paper
if type(dataset) == BinaryDbReaderSTB:

    import matplotlib.pyplot as plt
    curve_list = get_stb_ref_curves()
    curve_list.append((threshs, pck_curve_all, 'Ours (AUC=%.3f)' % auc_subset))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for t, v, name in curve_list:
        ax.plot(t, v, label=name)
    ax.set_xlabel('threshold in mm')
    ax.set_ylabel('PCK')
    plt.legend(loc='lower right')
    plt.savefig('eval_e2e.png')
    plt.show()
