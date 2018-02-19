import tensorflow as tf
import math
import pickle, json, os
import numpy as np
import numpy.linalg as nl
from utils.canonical_trafo import canonical_trafo, flip_right_hand
from utils.general import hand_size_tf, create_multiple_gaussian_map_3d, crop_image_from_xy

class DomeStreamReader(object):

    def __init__(self, mode='training', batch_size=1, shuffle=False, hand_crop=False, use_wrist_coord=False, crop_size_zoom=1.25, crop_size=256, sigma=25.0,
        coord_uv_noise=False, crop_center_noise=False, crop_offset_noise=False, crop_scale_noise=False, flip_2d=False):

        self.path_to_db = './data/DomeStreamTest1.pkl'

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hand_crop = hand_crop
        assert self.hand_crop
        self.use_wrist_coord = use_wrist_coord
        self.coord_uv_noise = coord_uv_noise
        self.sigma = sigma
        self.coord_uv_noise_sigma = 2.5  # std dev in px of noise on the uv coordinates
        self.crop_center_noise = crop_center_noise
        self.crop_center_noise_sigma = 20.0  # std dev in px: this moves what is in the "center", but the crop always contains all keypoints
        self.crop_offset_noise = crop_offset_noise
        self.crop_offset_noise_sigma = 10.0  # translates the crop after size calculation (this can move keypoints outside)
        self.crop_scale_noise = crop_scale_noise
        self.crop_size_zoom = crop_size_zoom
        self.flip_2d = flip_2d

        self.image_size = (1080, 1920)
        self.crop_size = crop_size

        with open(self.path_to_db, 'rb') as f:
            data = pickle.load(f)
        landmarks, img_dirs, calib_data = data

        for i in (1, 5, 9, 13, 17):
            landmarks[:, i:i+4, :] = landmarks[:, i+3:i-1:-1, :] # reverse the order of fingers (from palm to tip)

        K = np.array(calib_data['K'])
        K = np.expand_dims(K, axis=0)
        Ks = np.tile(K, (len(landmarks), 1, 1))

        self.num_samples = len(landmarks)
        self.Ks = tf.constant(Ks)
        self.joints2d = tf.constant(landmarks, dtype=tf.float32)
        self.hand_sides = tf.constant(np.zeros((len(landmarks),), dtype=bool))
        self.img_dirs = tf.constant(img_dirs)
        print('loaded Openpose data with number of samples {}'.format(self.num_samples))

    def get(self, read_image=True, extra=False):
        [joint2d, hand_side, img_dir, K] = tf.train.slice_input_producer([self.joints2d, self.hand_sides, self.img_dirs, self.Ks], shuffle=False)
        keypoint_uv21 = joint2d
        if not self.use_wrist_coord:
            palm_coord_uv = tf.expand_dims(0.5*(keypoint_uv21[0, :] + keypoint_uv21[12, :]), 0)
            keypoint_uv21 = tf.concat([palm_coord_uv, keypoint_uv21[1:21, :]], 0)
        if self.coord_uv_noise:
            noise = tf.truncated_normal([21, 2], mean=0.0, stddev=self.coord_uv_noise_sigma)
            keypoint_uv21 += noise

        data_dict = {}
        data_dict['img_dir'] = img_dir
        data_dict['hand_side'] = tf.one_hot(tf.cast(hand_side, tf.uint8), depth=2, on_value=1.0, off_value=0.0, dtype=tf.float32)
        data_dict["K"] = K

        keypoint_vis21 = tf.ones([21,], tf.bool)
        data_dict['keypoint_vis21'] = keypoint_vis21
        data_dict['keypoint_uv21_origin'] = data_dict['keypoint_uv21'] = keypoint_uv21

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
                crop_scale_noise = tf.squeeze(tf.exp(tf.truncated_normal([1], mean=0.0, stddev=0.1)))

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
            crop_size_best *= self.crop_size_zoom
            crop_size_best *= crop_scale_noise

            # calculate necessary scaling
            scale = tf.cast(self.crop_size, tf.float32) / crop_size_best
            # scale = tf.minimum(tf.maximum(scale, 1.0), 10.0)
            data_dict['crop_scale'] = scale
            if self.flip_2d:
                crop_center = tf.cond(hand_side, lambda: tf.constant(self.image_size, dtype=tf.float32) - tf.ones((2,), dtype=tf.float32)  - crop_center, lambda: crop_center)
            data_dict['crop_center'] = crop_center

            if self.crop_offset_noise:
                noise = tf.truncated_normal([2], mean=0.0, stddev=self.crop_offset_noise_sigma)
                crop_center += noise

            # Modify uv21 coordinates
            crop_center_float = tf.cast(crop_center, tf.float32)
            if self.flip_2d:
                keypoint_uv21_u = tf.cond(hand_side,
                    lambda: -(keypoint_uv21[:, 0] - crop_center_float[1]) * scale + self.crop_size // 2,
                    lambda: (keypoint_uv21[:, 0] - crop_center_float[1]) * scale + self.crop_size // 2)
            else:
                keypoint_uv21_u = (keypoint_uv21[:, 0] - crop_center_float[1]) * scale + self.crop_size // 2
            keypoint_uv21_v = (keypoint_uv21[:, 1] - crop_center_float[0]) * scale + self.crop_size // 2
            keypoint_uv21 = tf.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
            data_dict['keypoint_uv21'] = keypoint_uv21            

        if read_image:
            img_file = tf.read_file(img_dir)
            image = tf.image.decode_image(img_file, channels=3)
            image = tf.image.pad_to_bounding_box(image, 0, 0, 1080, 1920)
            image.set_shape((1080, 1920, 3))
            image = tf.cast(image, tf.float32)
            if self.hand_crop:
                img_crop = crop_image_from_xy(tf.expand_dims(image, 0), crop_center, self.crop_size, scale)
                img_crop =  img_crop / 255.0 - 0.5
                img_crop = tf.squeeze(img_crop)
                if self.flip_2d:
                    img_crop = tf.cond(hand_side, lambda: img_crop[:, ::-1, :], lambda: img_crop)
                data_dict['image_crop'] = img_crop
            if self.flip_2d:
                image = tf.cond(hand_side, lambda: image[:, ::-1, :], lambda: image)
            data_dict['image'] = image / 255.0 - 0.5

        keypoint_hw21 = tf.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], -1)

        scoremap_size = self.image_size
        
        if self.hand_crop:
            scoremap_size = (self.crop_size, self.crop_size)

        scoremap = self.create_multiple_gaussian_map(keypoint_hw21,
                                                     scoremap_size,
                                                     self.sigma,
                                                     valid_vec=keypoint_vis21,
                                                     extra=extra)

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
    def create_multiple_gaussian_map(coords_uv, output_size, sigma, valid_vec=None, extra=False):
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

            if extra:
                negative = 1 - tf.reduce_sum(scoremap, axis=2, keep_dims=True)
                negative = tf.minimum(tf.maximum(negative, 0.0), 1.0)
                scoremap = tf.concat([scoremap, negative], axis=2)

            return scoremap

if __name__ == '__main__':
    d = DomeStreamReader(mode='evaluation', flip_2d=True,
                             batch_size=1, shuffle=True, hand_crop=True, use_wrist_coord=True, crop_size=368, sigma=10.0,
                             crop_size_zoom=2.0, crop_center_noise=True, crop_offset_noise=True, crop_scale_noise=True)

    data = d.get(read_image=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    from utils.general import detect_keypoints_3d, plot_hand_3d, plot_hand, detect_keypoints
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    for i in range(50):
        image, image_crop, keypoint_uv21, img_dir, keypoint_uv21_origin, hand_side \
            = sess.run([data['image'], data['image_crop'], data['keypoint_uv21'], data['img_dir'], data['keypoint_uv21_origin'], data['hand_side']])
        print(img_dir[0].decode())
        print(hand_side)
        image_crop = np.squeeze((image_crop + 0.5) * 255).astype(np.uint8)
        image = np.squeeze((image + 0.5) * 255).astype(np.uint8)
        keypoint_uv21 = np.squeeze(keypoint_uv21)
        keypoint_uv21_origin = np.squeeze(keypoint_uv21_origin)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(image_crop)
        plot_hand(keypoint_uv21[:, ::-1], ax)
        ax = fig.add_subplot(122)
        ax.imshow(image)
        plot_hand(keypoint_uv21_origin[:, ::-1], ax)
        plt.show()
