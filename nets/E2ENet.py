import tensorflow as tf
from utils.general import NetworkOps
from nets.CPM import CPM
import os

ops = NetworkOps

class E2ENet(object):
    """ Network performing 3D pose estimation of a human hand from a single color image. """
    def __init__(self, lifting_dict, out_chan=21, crop_size=256):
        self.crop_size = crop_size
        self.out_chan = out_chan
        self.cpm = CPM(self.crop_size, self.out_chan)
        self.lifting_dict = lifting_dict
        assert lifting_dict['method'] in ['direct', 'heatmap']

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

    def inference(self, input_image, evaluation, train=False):
        heatmap_2d, encoding = self.cpm.inference(input_image, train)
        with tf.variable_scope("E2ENet"):
            scoremap = [encoding] + heatmap_2d
            with tf.variable_scope(self.lifting_dict['method']):
                scoremap = tf.concat(scoremap, axis=3)
                s = scoremap.get_shape().as_list()

                if self.lifting_dict['method'] == 'direct':
                    with tf.variable_scope('PosePrior'):
                        # some conv layers
                        out_chan_list = [32, 64, 128]
                        x = scoremap
                        for i, out_chan in enumerate(out_chan_list):
                            x = ops.conv_relu(x, 'conv_pose_%d_1' % i, kernel_size=3, stride=1, out_chan=out_chan, trainable=train)
                            x = ops.conv_relu(x, 'conv_pose_%d_2' % i, kernel_size=3, stride=2, out_chan=out_chan, trainable=train) # in the end this will be 4x4xC

                        # reshape and some fc layers
                        x = tf.reshape(x, [s[0], -1])
                        out_chan_list = [512, 512]
                        for i, out_chan in enumerate(out_chan_list):
                            x = ops.fully_connected_relu(x, 'fc_pose_%d' % i, out_chan=out_chan, trainable=train)
                            x = ops.dropout(x, 0.8, evaluation)

                        coord_xyz_can = ops.fully_connected(x, 'fc_xyz', out_chan=63, trainable=train)
                        coord_xyz_can = tf.reshape(coord_xyz_can, [s[0], 21, 3])

                    with tf.variable_scope('ViewPoint'):
                        x = scoremap
                        out_chan_list = [64, 128, 256]
                        for i, out_chan in enumerate(out_chan_list):
                            x = ops.conv_relu(x, 'conv_vp_%d_1' % i, kernel_size=3, stride=1, out_chan=out_chan, trainable=train)
                            x = ops.conv_relu(x, 'conv_vp_%d_2' % i, kernel_size=3, stride=2, out_chan=out_chan, trainable=train) # in the end this will be 4x4x128

                        # flatten
                        x = tf.reshape(x, [s[0], -1])  # this is Bx2048

                        # Estimate Viewpoint --> 3 params
                        out_chan_list = [256, 128]
                        for i, out_chan in enumerate(out_chan_list):
                            x = ops.fully_connected_relu(x, 'fc_vp_%d' % i, out_chan=out_chan, trainable=train)
                            x = ops.dropout(x, 0.75, evaluation)

                        ux = ops.fully_connected(x, 'fc_vp_ux', out_chan=1, trainable=train)
                        uy = ops.fully_connected(x, 'fc_vp_uy', out_chan=1, trainable=train)
                        uz = ops.fully_connected(x, 'fc_vp_uz', out_chan=1, trainable=train)

                    with tf.name_scope('get_rot_mat'):
                        u_norm = tf.sqrt(tf.square(ux) + tf.square(uy) + tf.square(uz) + 1e-8)
                        theta = u_norm

                        # some tmp vars
                        st_b = tf.sin(theta)
                        ct_b = tf.cos(theta)
                        one_ct_b = 1.0 - tf.cos(theta)

                        st = st_b[:, 0]
                        ct = ct_b[:, 0]
                        one_ct = one_ct_b[:, 0]
                        norm_fac = 1.0 / u_norm[:, 0]
                        ux = ux[:, 0] * norm_fac
                        uy = uy[:, 0] * norm_fac
                        uz = uz[:, 0] * norm_fac

                        rot_mat = self._stitch_mat_from_vecs([ct+ux*ux*one_ct, ux*uy*one_ct-uz*st, ux*uz*one_ct+uy*st,
                                                                   uy*ux*one_ct+uz*st, ct+uy*uy*one_ct, uy*uz*one_ct-ux*st,
                                                                   uz*ux*one_ct-uy*st, uz*uy*one_ct+ux*st, ct+uz*uz*one_ct])

                    coord_xyz_norm = tf.matmul(coord_xyz_can, rot_mat)

                    rel_dict = {'coord_xyz_norm': coord_xyz_norm, 'coord_xyz_can': coord_xyz_can, 'rot_mat': rot_mat, 'heatmap_2d': heatmap_2d}
                    return rel_dict

                elif self.lifting_dict['method'] == 'heatmap':
                    with tf.variable_scope('heatmap'):
                        assert s[1] == self.crop_size/8 and s[2] == self.crop_size/8
                        s3d = s[1]

                        x = ops.conv_relu(scoremap, 'lifting', kernel_size=1, stride=1, out_chan=s3d, trainable=train)
                        x = tf.transpose(x, perm=[0, 3, 1, 2])
                        encoding_3d = tf.expand_dims(x, -1)

                        x = ops.conv3d_relu(encoding_3d, 'conv3d1_stage1', kernel_size=5, stride=1, out_chan=128, leaky=False, trainable=train)
                        x = ops.conv3d_relu(x, 'conv3d2_stage1', kernel_size=5, stride=1, out_chan=128, leaky=False, trainable=train)
                        x = ops.conv3d_relu(x, 'conv3d3_stage1', kernel_size=5, stride=1, out_chan=128, leaky=False, trainable=train)
                        x = ops.conv3d_relu(x, 'conv3d4_stage1', kernel_size=1, stride=1, out_chan=128, leaky=False, trainable=train)
                        x = ops.conv3d(x, 'conv3d5_stage1', kernel_size=1, stride=1, out_chan=self.out_chan, trainable=train)

                        scoremap_3d = [x]

                        for stage_id in range(2, 4):
                            x = tf.concat([x, encoding_3d], axis=4)
                            x = ops.conv3d_relu(x, 'conv3d1_stage{}'.format(stage_id), kernel_size=5, stride=1, out_chan=128, leaky=False, trainable=train)
                            x = ops.conv3d_relu(x, 'conv3d2_stage{}'.format(stage_id), kernel_size=5, stride=1, out_chan=128, leaky=False, trainable=train)
                            x = ops.conv3d_relu(x, 'conv3d3_stage{}'.format(stage_id), kernel_size=5, stride=1, out_chan=128, leaky=False, trainable=train)
                            x = ops.conv3d_relu(x, 'conv3d4_stage{}'.format(stage_id), kernel_size=1, stride=1, out_chan=128, leaky=False, trainable=train)
                            x = ops.conv3d(x, 'conv3d5_stage{}'.format(stage_id), kernel_size=1, stride=1, out_chan=self.out_chan, trainable=train)
                            scoremap_3d.append(x)

                    rel_dict = {'heatmap_3d': scoremap_3d, 'heatmap_2d': heatmap_2d}
                    return rel_dict


    @staticmethod
    def _stitch_mat_from_vecs(vector_list):
        """ Stitches a given list of vectors into a 3x3 matrix.

            Input:
                vector_list: list of 9 tensors, which will be stitched into a matrix. list contains matrix elements
                    in a row-first fashion (m11, m12, m13, m21, m22, m23, m31, m32, m33). Length of the vectors has
                    to be the same, because it is interpreted as batch dimension.
        """

        assert len(vector_list) == 9, "There have to be exactly 9 tensors in vector_list."
        batch_size = vector_list[0].get_shape().as_list()[0]
        vector_list = [tf.reshape(x, [1, batch_size]) for x in vector_list]

        trafo_matrix = tf.dynamic_stitch([[0], [1], [2],
                                          [3], [4], [5],
                                          [6], [7], [8]], vector_list)

        trafo_matrix = tf.reshape(trafo_matrix, [3, 3, batch_size])
        trafo_matrix = tf.transpose(trafo_matrix, [2, 0, 1])

        return trafo_matrix
