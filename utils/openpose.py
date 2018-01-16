import tensorflow as tf
import numpy as np

def random_rotate(joint3d, slight):
    """ This function randomly rotates the hand in the space.
    """
    assert type(slight) == bool
    if slight:
        min_ = -np.pi/12
        max_ = np.pi/12
    else:
        min_ = 0.0
        max_ = 2*np.pi
    with tf.name_scope('random_rotate'):
        root = joint3d[0, :]
        root_relative = joint3d - root
        x = tf.random_uniform((1, ), minval=min_, maxval=max_)
        y = tf.random_uniform((1, ), minval=min_, maxval=max_)
        z = tf.random_uniform((1, ), minval=min_, maxval=max_)
        cx = tf.cos(x)
        sx = tf.sin(x)
        cy = tf.cos(y)
        sy = tf.sin(y)
        cz = tf.cos(z)
        sz = tf.sin(z)
        
        rx = tf.dynamic_stitch([[0], [1], [2],
                                      [3], [4], [5],
                                      [6], [7], [8]], [[1.0], [0.0], [0.0],
                                                       [0.0], cx,    -sx,
                                                       [0.0], sx,    cx])
        rx = tf.reshape(rx, [3, 3])

        ry = tf.dynamic_stitch([[0], [1], [2],
                              [3], [4], [5],
                              [6], [7], [8]], [cy,    [0.0], -sy, 
                                               [0.0], [1.0], [0.0],
                                               sy,    [0.0], cy])
        ry = tf.reshape(ry, [3, 3])

        rz = tf.dynamic_stitch([[0], [1], [2],
                              [3], [4], [5],
                              [6], [7], [8]], [cz,    -sz,   [0.0], 
                                               sz,    cz,    [0.0],
                                               [0.0], [0.0], [1.0]])
        rz = tf.reshape(rz, [3, 3])

        rotation = tf.matmul(tf.matmul(rz, ry), rx)
        rotated_joints = tf.matmul(root_relative, rotation) + root

    return rotated_joints

def project_tf(joint3d, calibK, calibR, calibt):
    """ This function projects the 3D hand to 2D using camera parameters
    """
    with tf.name_scope('project_tf'):
        x = tf.matmul(joint3d, calibR, transpose_b=True) + calibt
        xp = tf.transpose(tf.stack([tf.divide(x[:, 0], x[:, 2]), tf.divide(x[:, 1], x[:, 2])], axis=0))
        pt = tf.matmul(xp, calibK[:2, :2], transpose_b=True) + calibK[:2, 2]
    return pt, x