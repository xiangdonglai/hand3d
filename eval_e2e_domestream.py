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
import argparse, cv2, os, json
import pdb

from data.DomeStreamReader import DomeStreamReader
from nets.E2ENet import E2ENet
from utils.general import EvalUtil, get_stb_ref_curves, calc_auc, plot_hand_3d, detect_keypoints, trafo_coords, plot_hand, detect_keypoints_3d, hand_size, load_weights_from_snapshot
from utils.wrapper_hand_model import wrapper_hand_model
from utils.camera import project

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', '-v', action='store_true')
parser.add_argument('--save', '-s', action='store_true')
parser.add_argument('--output_dir', '-o', type=str, default='/home/donglaix/Documents/Experiments/output3d_domestream_fixcoeff')
args = parser.parse_args()

wrapper = wrapper_hand_model()

# get dataset
dataset = DomeStreamReader(mode='evaluation', shuffle=False, use_wrist_coord=True, hand_crop=True, crop_size=368, crop_size_zoom=2.0, flip_2d=True)

# build network graph
data = dataset.get()
image_crop = data['image_crop']
# lifting_dict = {'method': 'direct'}
lifting_dict = {'method': 'heatmap'}

# build network
net = E2ENet(lifting_dict, out_chan=22, crop_size=368)

# feed through network
evaluation = tf.placeholder_with_default(True, shape=())
rel_dict = net.inference(image_crop, evaluation, train=False)
data['heatmap_3d'] = rel_dict['heatmap_3d'][-1]
s = data['image_crop'].get_shape().as_list()
heatmap_2d = tf.image.resize_images(rel_dict['heatmap_2d'][-1], (s[1], s[2]))
data['heatmap_2d'] = heatmap_2d

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

# cpt = 'snapshots_e2e/model-100000'
# cpt = 'snapshots_e2e_RHD_STB_nw/model-38000'
cpt = 'snapshots_e2e_heatmap/model-100000'
load_weights_from_snapshot(sess, cpt, discard_list=['Adam', 'global_step', 'beta', 'scale'])

names = ['keypoint_uv21', 'image', 'image_crop', 'crop_scale', 'crop_center', 'heatmap_2d', 'K']

for i in range(dataset.num_samples):
    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

    # get prediction
    if lifting_dict['method'] == 'direct':
        names.append('coord_xyz_norm')
    elif lifting_dict['method'] == 'heatmap':
        names.append('heatmap_3d')
    items = [data[_] for _ in names]
    values = sess.run(items)
    value_dict = {x: y for x, y in zip(names, values)}

    if i >= 0:
        image_crop_v = np.squeeze((value_dict['image_crop']+0.5)*255).astype(np.uint8)
        image_v = np.squeeze((value_dict['image']+0.5)*255).astype(np.uint8)

        if lifting_dict['method'] == 'direct':
            coord3d_pred_v = np.squeeze(value_dict['coord_xyz_norm'])
        elif lifting_dict['method'] == 'heatmap':
            heatmap_3d_v = np.squeeze(value_dict['heatmap_3d'])
            coord3d_pred_v = detect_keypoints_3d(heatmap_3d_v)
            coord3d_pred_v = coord3d_pred_v[:21, :]

        coord3d_pred_v -= coord3d_pred_v[0, :]
        coord3d_pred_v /= hand_size(coord3d_pred_v)
        coord3d_pred_v *= 0.7

        coord2d_hw_v = detect_keypoints(np.squeeze(value_dict['heatmap_2d']))[:21, :]
        crop_scale = np.squeeze(value_dict['crop_scale'])
        crop_center = np.squeeze(value_dict['crop_center'])
        K = np.squeeze(value_dict['K'])

        wrapper.reset_value()

        coord3d_rev = (coord3d_pred_v + np.array([[0.0, 0.0, 10.0]])) * 100
        for ij in (1, 5, 9, 13, 17):
            coord3d_rev[ij:ij+4] = coord3d_rev[ij+3:ij-1:-1]
        wrapper.fit3d(coord3d_rev)

        coord2d_hw_global = (coord2d_hw_v - dataset.crop_size/2) / crop_scale + crop_center
        coord2d_uv_global = np.copy(coord2d_hw_global[:, ::-1])
        for ij in (1, 5, 9, 13, 17):
            coord2d_uv_global[ij:ij+4] = coord2d_uv_global[ij+3:ij-1:-1]
        coord3d_fit = wrapper.fit2d(coord2d_uv_global, K)
        kp2d_uv, _ = project(coord3d_fit, K)
        for ij in (1, 5, 9, 13, 17):
            kp2d_uv[ij:ij+4] = kp2d_uv[ij+3:ij-1:-1]

        glimg1 = wrapper.render(cameraMode=True, target=False, first_render=True)
        mask = (np.sum(glimg1, axis=2, keepdims=True) == 0.0)
        image_overlap = np.tile(mask, (1,1,3)) * image_v + glimg1

        concat = np.array(image_overlap, dtype=np.uint8)
        concat = cv2.resize(concat, (1066, 600))

        glimg2 = wrapper.render(cameraMode=False, target=False, first_render=True)
        concat = np.concatenate((concat, np.array(glimg2, dtype=np.uint8)), axis=1)

        glimg3 = wrapper.render(cameraMode=False, target=False, first_render=False, position=1)
        concat = np.concatenate((concat, np.array(glimg3, dtype=np.uint8)), axis=1)

        glimg4 = wrapper.render(cameraMode=False, target=False, first_render=False, position=2)
        concat = np.concatenate((concat, np.array(glimg4, dtype=np.uint8)), axis=1)

        assert cv2.imwrite(os.path.join(args.output_dir, '{:04d}.png'.format(i)), concat[:, :, ::-1])

        # fig = plt.figure()
        # ax1 = fig.add_subplot(151)
        # ax1.imshow(image_v)
        # plot_hand(coord2d_hw_global, ax1, color_fixed=np.array([1.0, 0.0, 0.0]))
        # plot_hand(kp2d_uv[:, ::-1], ax1, color_fixed=np.array([0.0, 1.0, 0.0]))
        # ax2 = fig.add_subplot(152)
        # ax2.imshow(image_overlap)
        # ax3 = fig.add_subplot(153)
        # ax3.imshow(glimg2)
        # ax4 = fig.add_subplot(154)
        # ax4.imshow(glimg3)
        # ax5 = fig.add_subplot(155)
        # ax5.imshow(glimg4)

        # plt.show()
