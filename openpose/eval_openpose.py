import tensorflow as tf
from openpose.OpenposeReader import OpenposeReader
from nets.PosePriorNetwork import PosePriorNetwork
import cv2, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utils.general import plot_hand_3d, load_weights_from_snapshot, detect_keypoints, plot_hand

input_dir = '/home/donglaix/Documents/Experiments/detected_hand1/'
output_dir = '/home/donglaix/Documents/Experiments/output_3d_freiburg1/'

dataset = OpenposeReader(mode='evaluation', shuffle=False, hand_crop=True, use_wrist_coord=False)
data = dataset.get()

net = PosePriorNetwork('proposed')
evaluation = tf.placeholder_with_default(True, shape=())
coord3d_pred, coord3d, _ = net.inference(data['scoremap'], data['hand_side'], evaluation)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

# net.init(sess, weight_files=['./weights/lifting-proposed-dome-hs.pickle'], exclude_var_list=['scale'])
last_cpt = tf.train.latest_checkpoint('./snapshots_lifting_proposed_dome_hs/')
assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network?"
load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta', 'scale'])

for i in range(dataset.num_samples):
    coord3d_pred_v, key, scoremap_v = sess.run([coord3d_pred, data['key'], data['scoremap']])
    coord3d_pred_v = np.squeeze(coord3d_pred_v)
    scoremap_v = np.squeeze(scoremap_v)
    key = key[0].decode()

    coord3d_pred_v[0, :] = 2 * coord3d_pred_v[0, :] - coord3d_pred_v[12, :]
    coord3d_pred_v -= coord3d_pred_v[0, :]

    filename = 'handTest_' + key + '_rendered.png'
    img = os.path.join(input_dir, filename)
    print(img)
    img = cv2.imread(img)
    assert img is not None

    fig = plt.figure(filename)
    ax1 = fig.add_subplot(111, projection='3d')
    plot_hand_3d(coord3d_pred_v * 15, ax1, color_fixed=np.array([1.0, 0.0, 0.0]))
    plt.xlabel('x')
    plt.ylabel('y')
    ax1.view_init(azim=-90.0, elev=-70.0)
    ax1.set_xlim(-3., 3.)
    ax1.set_ylim(-3., 3.)
    ax1.set_zlim(-3., 3.)

    # ax2 = fig.add_subplot(122)
    # keypoints2d = detect_keypoints(scoremap_v)
    # plot_hand(keypoints2d, ax2, color_fixed=np.array([1.0, 0.0, 0.0]))
    # ax2.set_xlim(0.0, 256.0)
    # ax2.set_ylim(0.0, 256.0)
    # ax2.invert_yaxis()
    # plt.show()

    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    fig.canvas.draw()
    figure = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    figure = figure.reshape(fig.canvas.get_width_height()[::-1] + (3,))[:, :, ::-1]

    plt.close()

    dh = figure.shape[0]
    dw = int(float(img.shape[1]) / img.shape[0] * dh)
    resized_img = cv2.resize(img, (dw, dh))

    concat = np.concatenate((resized_img, figure), axis=1)
    output_file = os.path.join(output_dir, filename)
    cv2.imwrite(output_file, concat)
