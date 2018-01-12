import tensorflow as tf
import pickle
import numpy as np

class OpenposeReader(object):
    def __init__(self, mode='evaluation', batch_size=1, shuffle=False, hand_crop=False, use_wrist_coord=False):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hand_crop = hand_crop
        assert self.hand_crop
        self.use_wrist_coord = use_wrist_coord
        self.sigma = 25.0

        self.path_to_db = './openpose/openpose_data.pkl'
        with open(self.path_to_db, 'rb') as f:
            data = pickle.load(f)

        assert mode == 'evaluation'
        self.image_size = (720, 1280)
        self.crop_size = 256

        keys = []
        joints2d = []
        for key, joint2d in data:
            joints2d.append(joint2d)
            keys.append(key)

        keys = np.array(keys)
        joints2d = np.array(joints2d, dtype=np.float32)

        for i in (1, 5, 9, 13, 17):
            joints2d[:, i:i+4, :] = joints2d[:, i+3:i-1:-1, :] # reverse the order of fingers (from palm to tip)

        self.num_samples = len(joints2d)
        self.keys = keys
        self.joints2d = joints2d

    def get(self):
        [joint2d, key] = tf.train.slice_input_producer([self.joints2d, self.keys], shuffle=False)
        keypoint_uv21 = joint2d
        if not self.use_wrist_coord:
            palm_coord_uv = tf.expand_dims(0.5*(keypoint_uv21[0, :] + keypoint_uv21[12, :]), 0)
            keypoint_uv21 = tf.concat([palm_coord_uv, keypoint_uv21[1:21, :]], 0)

        hand_side = 1
        # in Hanbyul's video, all are right hand
        data_dict = {'key': key}
        data_dict['hand_side'] = tf.one_hot(tf.cast(hand_side, tf.uint8), depth=2, on_value=1.0, off_value=0.0, dtype=tf.float32)

        keypoint_vis21 = tf.ones([21,], tf.bool)
        data_dict['keypoint_vis21'] = keypoint_vis21
        data_dict['keypoint_uv21'] = keypoint_uv21

        if self.hand_crop:
            crop_center = keypoint_uv21[12, ::-1]
            crop_center = tf.cond(tf.reduce_all(tf.is_finite(crop_center)), lambda: crop_center,
                                  lambda: tf.constant([0.0, 0.0]))
            crop_center.set_shape([2,])

            crop_scale_noise = tf.constant(1.0)

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
    d = OpenposeReader(mode='evaluation', batch_size=1, shuffle=True, hand_crop=True, use_wrist_coord=False)
    data = d.get()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
