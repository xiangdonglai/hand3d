import tensorflow as tf
import math
import pickle, json
import numpy as np
import numpy.linalg as nl
from utils.canonical_trafo import canonical_trafo, flip_right_hand
from utils.general import crop_image_from_xy
import os

class TsimonDBReader(object):

    def __init__(self, mode='training', batch_size=1, shuffle=False, hand_crop=False, use_wrist_coord=False, random_hue=False, random_rotate=False,
        coord_uv_noise=False, crop_center_noise=False, crop_offset_noise=False, crop_scale_noise=False, crop_size=256, sigma=25.0):

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
        self.random_hue = random_hue
        self.random_rotate = random_rotate

        self.image_size = (1080, 1920)
        self.crop_size = crop_size

        self.image_root = '/media/posefs0c/Users/donglaix/tsimon/'
        assert os.path.exists(self.image_root)
        self.path_to_db = ['/media/posefs0c/Users/donglaix/tsimon/hands_v12.json', '/media/posefs0c/Users/donglaix/tsimon/hands_v13.json', '/media/posefs0c/Users/donglaix/tsimon/hands_v143.json']
        assert mode == 'training'

        # Building a list of tensors
        joints2d = []
        hand_sides = []
        img_dirs = []
        vis = []

        for filename in self.path_to_db:
            with open(filename) as f:
                filedata = json.load(f)
            for ihand, hand_data in enumerate(filedata['root']):
                joint2d = np.array(hand_data['joint_self'])
                for i in (1, 5, 9, 13, 17):
                    joint2d[i:i+4] = joint2d[i+3:i-1:-1] # reverse the order of fingers (from palm to tip)
                joints2d.append(joint2d[:, :2])
                vis.append(joint2d[:, 2])
                hand_sides.append(1) # all are right hands
                img_dir = '{}/{}'.format(self.image_root, '/'.join(hand_data['img_paths'].split('/')[5:]))
                img_dirs.append(img_dir)
        assert len(img_dirs) > 0

        self.num_samples = len(joints2d)
        joints2d = np.array(joints2d, dtype=np.float32)
        hand_sides = np.array(hand_sides, dtype=bool)
        img_dirs = np.array(img_dirs)
        vis = np.array(vis, dtype=bool)

        self.joints2d = tf.constant(joints2d) # 94221, 21, 2
        self.hand_sides = tf.constant(hand_sides) # 94221, 
        self.img_dirs = tf.constant(img_dirs)
        self.vis = tf.constant(vis)
        print('loaded TsimonDB with number of samples {}'.format(self.num_samples))

    def get(self, read_image=False, extra=False):

        [joint2d, hand_side, img_dir, vis] = tf.train.slice_input_producer([self.joints2d, self.hand_sides, self.img_dirs, self.vis], shuffle=self.shuffle)
        keypoint_uv21 = joint2d
        keypoint_vis21 = vis

        if not self.use_wrist_coord:
            palm_coord_uv = tf.expand_dims(0.5*(keypoint_uv21[0, :] + keypoint_uv21[12, :]), 0)
            keypoint_uv21 = tf.concat([palm_coord_uv, keypoint_uv21[1:21, :]], 0)
            palm_vis = tf.expand_dims(tf.logical_or(keypoint_vis21[0], keypoint_vis21[12]), 0)
            keypoint_vis21 = tf.concat([palm_vis, keypoint_vis21[1:21]], 0)

        if self.coord_uv_noise:
            noise = tf.truncated_normal([21, 2], mean=0.0, stddev=self.coord_uv_noise_sigma)
            keypoint_uv21 += noise

        data_dict = {}
        data_dict['img_dir'] = img_dir
        data_dict['hand_side'] = tf.one_hot(tf.cast(hand_side, tf.uint8), depth=2, on_value=1.0, off_value=0.0, dtype=tf.float32)

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

            # select visible coords only
            kp_coord_h = tf.boolean_mask(keypoint_uv21[:, 1], keypoint_vis21)
            kp_coord_w = tf.boolean_mask(keypoint_uv21[:, 0], keypoint_vis21)
            kp_coord_hw = tf.stack([kp_coord_h, kp_coord_w], 1)

            # determine size of crop (measure spatial extend of hw coords first)
            # min_coord = tf.maximum(tf.reduce_min(kp_coord_hw, 0), 0.0)
            # max_coord = tf.minimum(tf.reduce_max(kp_coord_hw, 0), self.image_size)
            min_coord = tf.reduce_min(kp_coord_hw, 0)
            max_coord = tf.reduce_max(kp_coord_hw, 0)

            # find out larger distance wrt the center of crop
            crop_size_best = 2*tf.maximum(max_coord - crop_center, crop_center - min_coord)
            crop_size_best = tf.reduce_max(crop_size_best)
            crop_size_best = tf.minimum(tf.maximum(crop_size_best, 50.0), 500.0)
            crop_size_best = tf.cond(tf.reduce_all(tf.is_finite(crop_size_best)), lambda: crop_size_best,
                                  lambda: tf.constant(200.0))
            crop_size_best.set_shape([])
            data_dict['head_size'] = crop_size_best
            crop_size_best *= 2.0

            crop_scale_noise = tf.constant(1.0)
            if self.crop_scale_noise:
                crop_scale_noise = tf.squeeze(tf.random_uniform([1], minval=0.8, maxval=1.2)) 
            crop_size_best *= crop_scale_noise

            # calculate necessary scaling
            scale = tf.cast(self.crop_size, tf.float32) / crop_size_best
            # scale = tf.minimum(tf.maximum(scale, 1.0), 10.0)
            # scale *= crop_scale_noise
            data_dict['crop_scale'] = scale
            data_dict['crop_center'] = crop_center

            # Modify uv21 coordinates
            crop_center_float = tf.cast(crop_center, tf.float32)
            keypoint_uv21_u = tf.cond(hand_side,
                lambda: -(keypoint_uv21[:, 0] - crop_center_float[1]) * scale + self.crop_size // 2,
                lambda: (keypoint_uv21[:, 0] - crop_center_float[1]) * scale + self.crop_size // 2)
            keypoint_uv21_v = (keypoint_uv21[:, 1] - crop_center_float[0]) * scale + self.crop_size // 2
            keypoint_uv21 = tf.stack([keypoint_uv21_u, keypoint_uv21_v], 1)

            if self.random_rotate:
                angle = tf.random_uniform([], minval=-np.pi*80/180, maxval=np.pi*80/180)
                rotate_matrix = tf.dynamic_stitch([[0], [1], [2], [3]], [[tf.cos(angle)], [-tf.sin(angle)], [tf.sin(angle)], [tf.cos(angle)]])
                rotate_matrix = tf.reshape(rotate_matrix, [2, 2])
                keypoint_uv21 = tf.matmul(keypoint_uv21 - self.crop_size//2, rotate_matrix) + self.crop_size//2
            data_dict['keypoint_uv21'] = keypoint_uv21            

        if read_image:
            img_file = tf.read_file(img_dir)
            image = tf.image.decode_image(img_file, channels=3)
            image = tf.image.pad_to_bounding_box(image, 0, 0, 1080, 1920)
            image.set_shape((1080, 1920, 3))
            image = tf.cast(image, tf.float32)
            img_crop = crop_image_from_xy(tf.expand_dims(image, 0), crop_center, self.crop_size, scale)
            img_crop = tf.squeeze(img_crop)
            # return left hand only: flip the cropped image if hand_side is True.
            img_crop = tf.cond(hand_side, lambda: img_crop[:, ::-1, :], lambda: img_crop)
            if self.random_rotate:
                img_crop = tf.contrib.image.rotate(img_crop, angle)
            img_crop = img_crop / 255.0 - 0.5

            if self.random_hue:
                img_crop = tf.image.random_hue(img_crop, 0.1)
            data_dict['image_crop'] = img_crop

        keypoint_hw21 = tf.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], -1)

        assert self.hand_crop
        if self.hand_crop:
            scoremap_size = (self.crop_size, self.crop_size)

        scoremap = self.create_multiple_gaussian_map(keypoint_hw21,
                                                     scoremap_size,
                                                     self.sigma,
                                                     valid_vec=keypoint_vis21, extra=extra)

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
    d = TsimonDBReader(mode='training',
                         batch_size=1, shuffle=True, hand_crop=True, use_wrist_coord=True, crop_size=368, random_hue=False, random_rotate=True,
                         crop_center_noise=True, crop_offset_noise=True, crop_scale_noise=True)
    data = d.get(read_image=True)
    resized_ = tf.image.resize_images(data['scoremap'], [48, 48])
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    from utils.general import detect_keypoints, plot_hand
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    for i in range(50):
        scoremap, keypoint_uv21, image_crop, resized, img_dir, keypoint_vis21 \
            = sess.run([data['scoremap'], data['keypoint_uv21'], data['image_crop'], resized_, data['img_dir'], data['keypoint_vis21']])
        scoremap = np.squeeze(scoremap)
        resized = np.squeeze(resized)
        image_crop = np.squeeze((image_crop+0.5) * 255).astype(np.uint8)
        keypoint_uv21 = np.squeeze(keypoint_uv21)
        keypoint_vis21 = np.squeeze(keypoint_vis21)

        keypoints = detect_keypoints(scoremap)
        resized_keypoints = detect_keypoints(resized)
        print(img_dir[0].decode())
        # print(keypoint_vis21)
        # print(keypoint_uv21)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        plot_hand(keypoints, ax)
        plt.imshow(image_crop)
        ax = fig.add_subplot(122)
        plot_hand(keypoint_uv21[:, ::-1], ax)
        plt.imshow(image_crop)
        plt.show()