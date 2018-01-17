import tensorflow as tf
from utils.general import NetworkOps

ops = NetworkOps

class E2ENet(object):
    """ Network performing 3D pose estimation of a human hand from a single color image. """
    def __init__(self, output_size):
        self.crop_size = 256
        self.output_size = output_size
        self.num_kp = 21

    def init(self, session, weight_files=None, exclude_var_list=None):
        """ Initializes weights from pickled python dictionaries.

            Inputs:
                session: tf.Session, Tensorflow session object containing the network graph
                weight_files: list of str, Paths to the pickle files that are used to initialize network weights
                exclude_var_list: list of str, Weights that should not be loaded
        """
        if exclude_var_list is None:
            exclude_var_list = list()

        import pickle

        assert weight_files is not None

        # Initialize with weights
        for file_name in weight_files:
            assert os.path.exists(file_name), "File not found."
            with open(file_name, 'rb') as fi:
                weight_dict = pickle.load(fi)
                weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in exclude_var_list])}
                if len(weight_dict) > 0:
                    init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
                    session.run(init_op, init_feed)
                    print('Loaded %d variables from %s' % (len(weight_dict), file_name))


    def inference(self, input_image, train=False):
        with tf.variable_scope("E2ENet"):
            s = input_image.get_shape().as_list()
            assert s[1] == self.crop_size and s[2] == self.crop_size
            layers_per_block = [2, 2, 4, 2]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            # learn some feature representation, that describes the image content well
            x = input_image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            x = ops.conv_relu(x, 'conv4_3', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_4', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_5', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_6', kernel_size=3, stride=1, out_chan=256, trainable=train)
            encoding = ops.conv(x, 'conv4_7', kernel_size=3, stride=1, out_chan=32, trainable=train)

            x = ops.conv_relu(encoding, 'conv5_1', kernel_size=3, stride=1, out_chan=32, trainable=train)
            x = ops.conv_relu(x, 'conv5_2', kernel_size=3, stride=1, out_chan=32, trainable=train)
            scoremap_2d = ops.conv(x, 'conv5_3', kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)

            encoding_concat = tf.concat([encoding, scoremap_2d], axis=3)
            x = ops.conv_relu(encoding_concat, 'conv5_4', kernel_size=3, stride=1, out_chan=32, trainable=train)

            s = x.get_shape().as_list()
            assert s[1] == self.output_size and s[2] == self.output_size and s[3] == self.output_size

            x = tf.transpose(x, perm=[0, 3, 1, 2])
            x = tf.expand_dims(x, -1)

            layers_per_block = [2, 2, 4, 4]
            out_chan_list = [2, 4, 8, 16]
            for block_id, (layer_num, chan_num) in enumerate(zip(layers_per_block, out_chan_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv3d_relu(x, 'conv{}_{}'.format(block_id+6, layer_id), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)

            x = ops.conv3d(x, 'conv10', kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)

        return x, scoremap_2d