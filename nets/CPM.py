import tensorflow as tf
from utils.general import NetworkOps
import numpy as np

ops = NetworkOps

class CPM(object):
    # The original CPM: set input image to right hand, BGR channel order (OpenCV), image scale to x / 256.0 - 0.5, output channel number to 22 (the last one for background)

    def __init__(self, crop_size=256, out_chan=21):
        self.out_chan = out_chan
        self.crop_size = crop_size

    def init(self, weight_path, sess):
        with tf.variable_scope("CPM"):
            data_dict = np.load(weight_path, encoding='latin1').item()
            for op_name in data_dict:
                with tf.variable_scope(op_name, reuse=True):
                    for param_name, data in data_dict[op_name].items():
                        var = tf.get_variable(param_name)
                        sess.run(var.assign(data))
        print('Finish loading weight from {}'.format(weight_path))

    def init_vgg(self, sess, weight_path='./weights/vgg16.npy'):
        with tf.variable_scope("CPM"):
            data_dict = np.load(weight_path, encoding='latin1').item()
            for op_name in data_dict:
                if not op_name.startswith("conv") or op_name == 'conv5_3':
                    continue
                with tf.variable_scope(op_name, reuse=True):
                    assert len(data_dict[op_name]) == 2
                    for data in data_dict[op_name]:
                        if data.ndim == 4:
                            var = tf.get_variable('weights')
                        elif data.ndim == 1:
                            var = tf.get_variable('biases')
                        else:
                            raise Exception
                        sess.run(var.assign(data))
        print('Finish loading weight from {}'.format(weight_path))

    def inference(self, input_image, train=False):
        with tf.variable_scope("CPM"):
            s = input_image.get_shape().as_list()
            assert s[1] == self.crop_size and s[2] == self.crop_size

            layers_per_block = [2, 2, 4, 4, 2]
            out_chan_list = [64, 128, 256, 512, 512]
            pool_list = [True, True, True, False, False]

            # conv1_1 ~ conv4_4
            x = input_image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1, out_chan=chan_num, leaky=False, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            conv5_3 = ops.conv_relu(x, 'conv5_3_CPM', kernel_size=3, stride=1, out_chan=128, leaky=False, trainable=train)
            x = ops.conv_relu(conv5_3, 'conv6_1_CPM', kernel_size=1, stride=1, out_chan=512, leaky=False, trainable=train)
            x = ops.conv(x, 'conv6_2_CPM', kernel_size=1, stride=1, out_chan=self.out_chan, trainable=train)
            scoremaps = [x]

            for stage_id in range(2, 7):
                x = tf.concat([x, conv5_3], axis=3, name='concat_stage{}'.format(stage_id))
                for layer_id in range(1, 6):
                    x = ops.conv_relu(x, 'Mconv{}_stage{}'.format(layer_id, stage_id), kernel_size=7, stride=1, out_chan=128, leaky=False, trainable=train)
                x = ops.conv_relu(x, 'Mconv6_stage{}'.format(stage_id), kernel_size=1, stride=1, out_chan=128, leaky=False, trainable=train)
                x = ops.conv(x, 'Mconv7_stage{}'.format(stage_id), kernel_size=1, stride=1, out_chan=self.out_chan, trainable=train)
                scoremaps.append(x)

        return scoremaps, conv5_3