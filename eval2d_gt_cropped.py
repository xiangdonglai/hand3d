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
""" Script for isolated evaluation of PoseNet on hand cropped images.
    Ground truth keypoint annotations are used for crop generation.

    This allows to reproduce row 1 from Table 1 of the paper:
    GT R-val AUC=0.724 EPE median=5.001 EPE mean=9.135
"""
from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np
import argparse

from data.BinaryDbReader import *
from data.BinaryDbReaderSTB import *
from data.DomeReader import DomeReader
from data.ManualDBReader import ManualDBReader
from data.TsimonDBReader import TsimonDBReader
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from nets.CPM import CPM
from utils.general import detect_keypoints, EvalUtil, load_weights_from_snapshot, plot_hand
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', '-v', action='store_true')
parser.add_argument('--save', '-s', action='store_true')
args = parser.parse_args()

# flag that allows to load a retrained snapshot(original weights used in the paper are used otherwise)
USE_RETRAINED = False
FREIBURG_ORDER = False
PATH_TO_SNAPSHOTS = './snapshots_cpm_rotate_s10_wrist_dome_simon/'  # only used when USE_RETRAINED is true

# get dataset
# dataset = ManualDBReader(mode='evaluation', shuffle=False, hand_crop=True, use_wrist_coord=True, crop_size=368, crop_size_zoom=2.0)
dataset = DomeReader(mode='evaluation', shuffle=False, hand_crop=True, use_wrist_coord=True, crop_size=368, crop_size_zoom=2.0)

# build network graph
data = dataset.get(read_image=True)
# data = dataset.get()

# build network
evaluation = tf.placeholder_with_default(True, shape=())
net = CPM(crop_size=368, out_chan=22)
data['image_crop'] = data['image_crop'][:, :, :, ::-1] # convert to BGR (for tomas model)
keypoints_scoremap, _ = net.inference(data['image_crop'])
keypoints_scoremap = keypoints_scoremap[-1]

# upscale to original size
s = data['image_crop'].get_shape().as_list()
keypoints_scoremap = tf.image.resize_images(keypoints_scoremap, (s[1], s[2]))

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

# initialize network weights
if USE_RETRAINED:
    # retrained version
    last_cpt = tf.train.latest_checkpoint(PATH_TO_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    print('loading weights from {}'.format(last_cpt))
else:
    # load weights used in the paper
    net.init('./weights/pose_model.npy', sess)
    # net.init_pickle(sess, ['./snapshots_cpm_rotate_s10_wrist_dome/model-100000.pickle'], ['scale'])

util = EvalUtil()
# iterate dataset

results = []
for i in range(dataset.num_samples):
    # get prediction
    # crop_scale, keypoints_scoremap_v, kp_uv21_gt, kp_vis, image_crop, crop_center, img_dir, hand_side, head_size \
    #     = sess.run([data['crop_scale'], keypoints_scoremap, data['keypoint_uv21'], data['keypoint_vis21'], data['image_crop'], data['crop_center'], data['img_dir'], data['hand_side'], data['head_size']])
    crop_scale, keypoints_scoremap_v, kp_uv21_gt, kp_vis, image_crop, crop_center, img_dir, hand_side \
        = sess.run([data['crop_scale'], keypoints_scoremap, data['keypoint_uv21'], data['keypoint_vis21'], data['image_crop'], data['crop_center'], data['img_dir'], data['hand_side']])

    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    kp_uv21_gt = np.squeeze(kp_uv21_gt)
    kp_vis = np.squeeze(kp_vis)
    crop_scale = np.squeeze(crop_scale)
    crop_center = np.squeeze(crop_center)
    hand_side = np.squeeze(hand_side)
    # head_size = float(np.squeeze(head_size))

    # detect keypoints
    coord_hw_pred_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw_pred_crop = coord_hw_pred_crop[:21, :]
    if not FREIBURG_ORDER:
        for ik in (1, 5, 9, 13, 17):
            coord_hw_pred_crop[ik:ik+4] = coord_hw_pred_crop[ik+3:ik-1:-1] # reverse the order of fingers (from palm to tip)
    coord_uv_pred_crop = np.stack([coord_hw_pred_crop[:, 1], coord_hw_pred_crop[:, 0]], 1)

    image_crop = np.squeeze((image_crop+0.5)*255).astype(np.uint8)

    if not dataset.use_wrist_coord:
        kp_uv21_gt[0, :] = 2 * kp_uv21_gt[0, :] - kp_uv21_gt[12, :]
        coord_uv_pred_crop[0, :] = 2 * coord_uv_pred_crop[0, :] - coord_uv_pred_crop[12, :]
        coord_hw_pred_crop[0, :] = 2 * coord_hw_pred_crop[0, :] - coord_hw_pred_crop[12, :]
    util.feed(kp_uv21_gt/184, kp_vis, coord_uv_pred_crop/184)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

    if args.visualize:
        plt.imshow(image_crop)
        plot_hand(coord_hw_pred_crop, plt, color_fixed=np.array([0.0, 0.0, 1.0]))
        plot_hand(kp_uv21_gt[:, ::-1], plt, color_fixed=np.array([1.0, 0.0, 0.0]))
        plt.show()

    if args.save:
        result = {}
        result['img_dir'] = img_dir[0].decode()
        for i in (1, 5, 9, 13, 17):
            coord_uv_pred_crop[i:i+4] = coord_uv_pred_crop[i+3:i-1:-1] # reverse the order of fingers (from palm to tip)
        if int(hand_side[1]):
            coord_uv_pred_crop[:, 0] = net.crop_size - 1 - coord_uv_pred_crop[:, 0]
        coord_uv_pred_crop -= net.crop_size//2
        result['hand2d'] = (coord_uv_pred_crop/crop_scale + crop_center[::-1]).tolist()
        results.append(result)

mean, median, auc, pck_curve_all, threshs = util.get_measures(0.0, 1.0, 100)
print('Evaluation results:')
print('Average mean EPE: %.3f pixels' % mean)
print('Average median EPE: %.3f pixels' % median)
print('Area under curve: %.3f' % auc)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(threshs, pck_curve_all)
ax.set_xlabel('threshold')
ax.set_ylabel('PCK')
ax.set_xticks(np.arange(0, 1, 0.1))
ax.set_yticks(np.arange(0, 1, 0.1))
plt.grid()
plt.show()

if args.save:
    import json
    with open('./eval/detection_2d_mymeasure.json', 'w') as f:
        json.dump(results, f)
    import pickle
    with open('./eval/detection_2d_mymeasure.pkl', 'wb') as f:
        pickle.dump(util, f)