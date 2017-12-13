import numpy as np
from data.BinaryDbReader import BinaryDbReader
from utils.general import EvalUtil
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import pickle as cp
import os

if not os.path.isfile('training_data.pkl'):
    # dataset = BinaryDbReader(mode='evaluation', shuffle=False, hand_crop=True, use_wrist_coord=False)
    training_set = BinaryDbReader(mode='training', shuffle=False, hand_crop=True, use_wrist_coord=False)
    print('Finishing loading training set')

    training_data = training_set.get()
    train_data_uv21 = training_data['keypoint_uv21']
    train_data_xyz21 = training_data['keypoint_xyz21']

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(sess=sess)

    train_2d = []
    train_3d = []

    for i in range(training_set.num_samples):
        if i % 100 == 0:
            print('loading training hand {}/{}'.format(i+1, training_set.num_samples))
        train_uv21, train_xyz21 = sess.run([train_data_uv21, train_data_xyz21])
        train_uv21 = np.squeeze(train_uv21).reshape(-1)
        train_xyz21 = np.squeeze(train_xyz21)

        train_2d.append(train_uv21)
        train_3d.append(train_xyz21)

    sess.close()
    train_2d = np.array(train_2d)

    with open('training_data.pkl', 'wb') as f:
        cp.dump((train_2d, train_3d), f)

else:
    with open('training_data.pkl', 'rb') as f:
        train_2d, train_3d = cp.load(f)

if not os.path.isfile('testing_data.pkl'):
    dataset = BinaryDbReader(mode='evaluation', shuffle=False, hand_crop=True, use_wrist_coord=False)
    print('Finishing loading testing set')

    testing_data = dataset.get()
    test_data_uv21 = testing_data['keypoint_uv21']
    test_data_xyz21 = testing_data['keypoint_xyz21']

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(sess=sess)

    test_2d = []
    test_3d = []

    for i in range(dataset.num_samples):
        if i % 100 == 0:
            print('loading testing hand {}/{}'.format(i+1, dataset.num_samples))
        test_uv21, test_xyz21 = sess.run([test_data_uv21, test_data_xyz21])
        test_uv21 = np.squeeze(test_uv21).reshape(-1)
        test_xyz21 = np.squeeze(test_xyz21)

        test_2d.append(test_uv21)
        test_3d.append(test_xyz21)

    sess.close()
    test_2d = np.array(test_2d)

    with open('testing_data.pkl', 'wb') as f:
        cp.dump((test_2d, test_3d), f)

else:
    with open('testing_data.pkl', 'rb') as f:
        test_2d, test_3d = cp.load(f)

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(train_2d)
result = neigh.kneighbors(test_2d, return_distance=False)

util = EvalUtil()

for i in range(len(test_2d)):
    keypoint_xyz21 = test_3d[i]
    keypoint_xyz21 -= keypoint_xyz21[0, :]
    coord3d_pred_v = train_3d[result[i][0]]
    coord3d_pred_v -= coord3d_pred_v[0, :]
    kp_vis = np.ones_like(keypoint_xyz21[:, 0])
    util.feed(keypoint_xyz21, kp_vis, coord3d_pred_v)

mean, median, auc, _, _ = util.get_measures(0.0, 0.050, 20)
print('Average mean EPE: %.3f mm' % (mean*1000))
print('Average median EPE: %.3f mm' % (median*1000))