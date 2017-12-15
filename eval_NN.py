import numpy as np
from data.BinaryDbReader import BinaryDbReader
from utils.general import EvalUtil
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import pickle as cp
import os

if not os.path.isfile('rendered_training_data.pkl'):
    # dataset = BinaryDbReader(mode='evaluation', shuffle=False, hand_crop=True, use_wrist_coord=False)
    training_set = BinaryDbReader(mode='training', shuffle=False, hand_crop=True, use_wrist_coord=False)
    print('Finishing loading training set')

    training_data = training_set.get()
    train_data_uv21 = training_data['keypoint_uv21']
    train_data_xyz21 = training_data['keypoint_xyz21']
    train_data_lr = training_data['hand_side']

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(sess=sess)

    train = []

    for i in range(training_set.num_samples):
        data = {}
        if i % 100 == 0:
            print('loading training hand {}/{}'.format(i+1, training_set.num_samples))
        train_uv21, train_xyz21, train_lr = sess.run([train_data_uv21, train_data_xyz21, train_data_lr])
        train_uv21 = np.squeeze(train_uv21).reshape(-1)
        train_xyz21 = np.squeeze(train_xyz21)
        train_lr = int(np.squeeze(train_lr)[1])

        data['hand2d'] = train_uv21
        data['hand3d'] = train_xyz21
        data['lr'] = train_lr
        train.append(data)

    sess.close()

    with open('rendered_training_data.pkl', 'wb') as f:
        cp.dump(train, f, protocol=2)

else:
    with open('rendered_training_data.pkl', 'rb') as f:
        train = cp.load(f)

if not os.path.isfile('rendered_testing_data.pkl'):
    dataset = BinaryDbReader(mode='evaluation', shuffle=False, hand_crop=True, use_wrist_coord=False)
    print('Finishing loading testing set')

    testing_data = dataset.get()
    test_data_uv21 = testing_data['keypoint_uv21']
    test_data_xyz21 = testing_data['keypoint_xyz21']
    test_data_lr = testing_data['hand_side']

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(sess=sess)

    test = []

    for i in range(dataset.num_samples):
        data = {}
        if i % 100 == 0:
            print('loading testing hand {}/{}'.format(i+1, dataset.num_samples))
        test_uv21, test_xyz21, test_lr = sess.run([test_data_uv21, test_data_xyz21, test_data_lr])
        test_uv21 = np.squeeze(test_uv21).reshape(-1)
        test_xyz21 = np.squeeze(test_xyz21)
        test_lr = int(np.squeeze(test_lr)[1])

        data['hand2d'] = test_uv21
        data['hand3d'] = test_xyz21
        data['lr'] = test_lr
        test.append(data)

    sess.close()

    with open('rendered_testing_data.pkl', 'wb') as f:
        cp.dump(test, f, protocol=2)

else:
    with open('rendered_testing_data.pkl', 'rb') as f:
        test = cp.load(f)

train_2d = [_['hand2d'] for _ in train]
test_2d = [_['hand2d'] for _ in test]
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(train_2d)
result = neigh.kneighbors(test_2d, return_distance=False)

util = EvalUtil()

for i in range(len(test)):
    keypoint_xyz21 = test[i]['hand3d']
    keypoint_xyz21 -= keypoint_xyz21[0, :]
    coord3d_pred_v = train[result[i][0]]['hand3d']
    coord3d_pred_v -= coord3d_pred_v[0, :]
    kp_vis = np.ones_like(keypoint_xyz21[:, 0])
    util.feed(keypoint_xyz21, kp_vis, coord3d_pred_v)

mean, median, auc, _, _ = util.get_measures(0.0, 0.050, 20)
print('Average mean EPE: %.3f mm' % (mean*1000))
print('Average median EPE: %.3f mm' % (median*1000))