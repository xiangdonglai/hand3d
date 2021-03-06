import tensorflow as tf
import math
import pickle, json
import numpy as np
import numpy.linalg as nl
from utils.canonical_trafo import canonical_trafo, flip_right_hand
from utils.general import hand_size_tf
from utils.openpose import random_rotate, project_tf

class DomeAugReader(object):

    def __init__(self, mode='training', batch_size=1, shuffle=False, hand_crop=False, use_wrist_coord=False,
        coord_uv_noise=False, crop_center_noise=False, crop_offset_noise=False, crop_scale_noise=False, a4=False):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hand_crop = hand_crop
        assert self.hand_crop
        self.use_wrist_coord = use_wrist_coord
        self.coord_uv_noise = coord_uv_noise
        self.sigma = 25.0
        self.coord_uv_noise_sigma = 2.5  # std dev in px of noise on the uv coordinates
        self.crop_center_noise = crop_center_noise
        self.crop_center_noise_sigma = 20.0  # std dev in px: this moves what is in the "center", but the crop always contains all keypoints
        self.crop_offset_noise = crop_offset_noise
        self.crop_offset_noise_sigma = 10.0  # translates the crop after size calculation (this can move keypoints outside)
        self.crop_scale_noise = crop_scale_noise


        self.path_to_db = './data/hand_data.json'
        self.camera_file = './data/camera_data.pkl'
        with open(self.path_to_db) as f:
            json_data = json.load(f)
        with open(self.camera_file, 'rb') as f:
            self.camera_data = pickle.load(f, encoding='latin1')

        assert mode == 'training'
        training_data = json_data['training_data']

        if a4:
            self.path_to_db = './data/hand_data_a4.json'
            self.camera_file = './data/camera_data_a4.pkl'
            with open(self.path_to_db) as f:
                json_data_a4 = json.load(f)
            training_data += json_data_a4['training_data']

            with open(self.camera_file, 'rb') as f:
                camera_data_a4 = pickle.load(f, encoding='latin1')
            for key, value in camera_data_a4.items():
                assert key not in self.camera_data
                self.camera_data[key] = value

        self.image_size = (1080, 1920)
        self.crop_size = 256

        # Building a list of tensors
        joints3d = []
        calibKs = []
        calibRs = []
        calibts = []
        hand_sides = []
        for ihand, hand3d in enumerate(training_data):
            joint3d = np.array(hand3d['hand3d']).reshape(-1, 3)
            for i in (1, 5, 9, 13, 17):
                joint3d[i:i+4] = joint3d[i+3:i-1:-1] # reverse the order of fingers (from palm to tip)
            for camIdx in hand3d['hand2d']:
                seqName = hand3d['seqName']
                calib = self.camera_data[seqName][camIdx]
                calibK = calib['K']
                calibR = calib['R']
                calibt = calib['t'].ravel()
                joints3d.append(joint3d)
                calibKs.append(calibK)
                calibRs.append(calibR)
                calibts.append(calibt)
                hand_sides.append(hand3d['lr'])

        joints3d = np.array(joints3d, dtype=np.float32)
        calibRs = np.array(calibRs, dtype=np.float32)
        calibKs = np.array(calibKs, dtype=np.float32)
        calibts = np.array(calibts, dtype=np.float32)
        hand_sides = np.array(hand_sides, dtype=bool)

        self.num_samples = len(joints3d)
        self.joints3d = tf.constant(joints3d) # 94221, 21, 3
        self.calibRs = tf.constant(calibRs)
        self.calibKs = tf.constant(calibKs)
        self.calibts = tf.constant(calibts)
        self.hand_sides = tf.constant(hand_sides) # 94221, 

    def get(self, augment=True, slight=False):
        [joint3d, calibK, calibR, calibt, hand_side] = tf.train.slice_input_producer([self.joints3d, self.calibKs, self.calibRs, self.calibts, self.hand_sides], shuffle=False)
        if augment:
            # perform augment by rotation
            joint3d = random_rotate(joint3d, slight)
        joint2d, joint3d = project_tf(joint3d, calibK, calibR, calibt)
        keypoint_xyz21 = joint3d
        keypoint_uv21 = joint2d
        if not self.use_wrist_coord:
            palm_coord = tf.expand_dims(0.5*(keypoint_xyz21[0, :] + keypoint_xyz21[12, :]), 0)
            keypoint_xyz21 = tf.concat([palm_coord, keypoint_xyz21[1:21, :]], 0)
            palm_coord_uv = tf.expand_dims(0.5*(keypoint_uv21[0, :] + keypoint_uv21[12, :]), 0)
            keypoint_uv21 = tf.concat([palm_coord_uv, keypoint_uv21[1:21, :]], 0)
        if self.coord_uv_noise:
            noise = tf.truncated_normal([21, 2], mean=0.0, stddev=self.coord_uv_noise_sigma)
            keypoint_uv21 += noise

        data_dict = {}
        data_dict['hand_side'] = tf.one_hot(tf.cast(hand_side, tf.uint8), depth=2, on_value=1.0, off_value=0.0, dtype=tf.float32)

        keypoint_vis21 = tf.ones([21,], tf.bool)
        data_dict['keypoint_vis21'] = keypoint_vis21

        kp_coord_xyz21 = keypoint_xyz21
        data_dict['keypoint_xyz21'] = keypoint_xyz21
        data_dict['keypoint_uv21'] = keypoint_uv21
        kp_coord_xyz_root = kp_coord_xyz21[0, :] # this is the palm coord
        kp_coord_xyz21_rel = kp_coord_xyz21 - kp_coord_xyz_root  # relative coords in metric coords
        index_root_bone_length = tf.sqrt(tf.reduce_sum(tf.square(kp_coord_xyz21_rel[12, :] - kp_coord_xyz21_rel[11, :])))
        # data_dict['keypoint_scale'] = index_root_bone_length
        # data_dict['keypoint_xyz21_normed'] = kp_coord_xyz21_rel / index_root_bone_length  # normalized by length of 12->11
        data_dict['keypoint_scale'] = hand_size_tf(kp_coord_xyz21_rel)
        data_dict['keypoint_xyz21_normed'] = kp_coord_xyz21_rel / data_dict['keypoint_scale']

        # calculate viewpoint and coords in canonical coordinates
        cond_left = tf.logical_and(tf.cast(tf.ones_like(kp_coord_xyz21), tf.bool), tf.logical_not(hand_side))
        kp_coord_xyz21_rel_can, rot_mat = canonical_trafo(data_dict['keypoint_xyz21_normed'])
        kp_coord_xyz21_rel_can, rot_mat = tf.squeeze(kp_coord_xyz21_rel_can), tf.squeeze(rot_mat)
        kp_coord_xyz21_rel_can = flip_right_hand(kp_coord_xyz21_rel_can, tf.logical_not(cond_left))
        data_dict['keypoint_xyz21_can'] = kp_coord_xyz21_rel_can
        data_dict['rot_mat'] = tf.matrix_inverse(rot_mat)

        if self.hand_crop:
            crop_center = keypoint_uv21[12, ::-1]
            crop_center = tf.cond(tf.reduce_all(tf.is_finite(crop_center)), lambda: crop_center,
                                  lambda: tf.constant([0.0, 0.0]))
            crop_center.set_shape([2,])

            if self.crop_center_noise:
                noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_center_noise_sigma)
                crop_center += noise

            crop_scale_noise = tf.constant(1.0)
            if self.crop_scale_noise:
                crop_scale_noise = tf.squeeze(tf.random_uniform([1], minval=0.8, maxval=1.2))    

            kp_coord_hw = tf.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], 1)
            # determine size of crop (measure spatial extend of hw coords first)
            min_coord = tf.maximum(tf.reduce_min(kp_coord_hw, 0), 0.0)
            max_coord = tf.minimum(tf.reduce_max(kp_coord_hw, 0), self.image_size)

            # find out larger distance wrt the center of crop
            crop_size_best = 2*tf.maximum(max_coord - crop_center, crop_center - min_coord)
            crop_size_best = tf.reduce_max(crop_size_best)
            crop_size_best = tf.minimum(tf.maximum(crop_size_best, 50.0), 500.0)
            crop_size_best = tf.cond(tf.reduce_all(tf.is_finite(crop_size_best)), lambda: crop_size_best,
                                  lambda: tf.constant(200.0))
            crop_size_best.set_shape([])
            crop_size_best *= 1.25

            # calculate necessary scaling
            scale = tf.cast(self.crop_size, tf.float32) / crop_size_best
            scale = tf.minimum(tf.maximum(scale, 1.0), 10.0)
            scale *= crop_scale_noise
            data_dict['crop_scale'] = scale

            if self.crop_offset_noise:
                noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_offset_noise_sigma)
                crop_center += noise

            # Modify uv21 coordinates
            crop_center_float = tf.cast(crop_center, tf.float32)
            keypoint_uv21_u = (keypoint_uv21[:, 0] - crop_center_float[1]) * scale + self.crop_size // 2
            keypoint_uv21_v = (keypoint_uv21[:, 1] - crop_center_float[0]) * scale + self.crop_size // 2
            keypoint_uv21 = tf.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
            data_dict['keypoint_uv21'] = keypoint_uv21            


        keypoint_hw21 = tf.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], -1)

        scoremap_size = self.image_size
        
        if self.hand_crop:
            scoremap_size = (self.crop_size, self.crop_size)

        scoremap = self.create_multiple_gaussian_map(keypoint_hw21,
                                                     scoremap_size,
                                                     self.sigma,
                                                     valid_vec=keypoint_vis21)

        data_dict['scoremap'] = scoremap

        names, tensors = zip(*data_dict.items())

        if self.shuffle:
            tensors = tf.train.shuffle_batch_join([tensors],
                                                  batch_size=self.batch_size,
                                                  capacity=100,
                                                  min_after_dequeue=50,
                                                  enqueue_many=False)
        else:
            tensors = tf.train.batch_join([tensors],
                                          batch_size=self.batch_size,
                                          capacity=100,
                                          enqueue_many=False)

        return dict(zip(names, tensors))


    @staticmethod
    def create_multiple_gaussian_map(coords_uv, output_size, sigma, valid_vec=None):
        """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
            with variance sigma for multiple coordinates."""
        with tf.name_scope('create_multiple_gaussian_map'):
            sigma = tf.cast(sigma, tf.float32)
            assert len(output_size) == 2
            s = coords_uv.get_shape().as_list()
            coords_uv = tf.cast(coords_uv, tf.int32)
            if valid_vec is not None:
                valid_vec = tf.cast(valid_vec, tf.float32)
                valid_vec = tf.squeeze(valid_vec)
                cond_val = tf.greater(valid_vec, 0.5)
            else:
                cond_val = tf.ones_like(coords_uv[:, 0], dtype=tf.float32)
                cond_val = tf.greater(cond_val, 0.5)

            cond_1_in = tf.logical_and(tf.less(coords_uv[:, 0], output_size[0]-1), tf.greater(coords_uv[:, 0], 0))
            cond_2_in = tf.logical_and(tf.less(coords_uv[:, 1], output_size[1]-1), tf.greater(coords_uv[:, 1], 0))
            cond_in = tf.logical_and(cond_1_in, cond_2_in)
            cond = tf.logical_and(cond_val, cond_in)

            coords_uv = tf.cast(coords_uv, tf.float32)

            # create meshgrid
            x_range = tf.expand_dims(tf.range(output_size[0]), 1)
            y_range = tf.expand_dims(tf.range(output_size[1]), 0)

            X = tf.cast(tf.tile(x_range, [1, output_size[1]]), tf.float32)
            Y = tf.cast(tf.tile(y_range, [output_size[0], 1]), tf.float32)

            X.set_shape((output_size[0], output_size[1]))
            Y.set_shape((output_size[0], output_size[1]))

            X = tf.expand_dims(X, -1)
            Y = tf.expand_dims(Y, -1)

            X_b = tf.tile(X, [1, 1, s[0]])
            Y_b = tf.tile(Y, [1, 1, s[0]])

            X_b -= coords_uv[:, 0]
            Y_b -= coords_uv[:, 1]

            dist = tf.square(X_b) + tf.square(Y_b)

            scoremap = tf.exp(-dist / tf.square(sigma)) * tf.cast(cond, tf.float32)

            return scoremap

if __name__ == '__main__':
    d = DomeAugReader(mode='training',
                         batch_size=8, shuffle=True, hand_crop=True, use_wrist_coord=False,
                         coord_uv_noise=True, crop_center_noise=True, crop_offset_noise=True, crop_scale_noise=True)
    d.get()