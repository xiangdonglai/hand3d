import numpy as np
import numpy.linalg as nl
import json, math, pickle
from utils.general import hand_size, plot_hand_3d
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train = True
test = True

def canonical_3d(keypoint3d, flip):
    """
    This function calculate the canonical hand pose in 3D space.
    The wrist is at the original point; the end of middle finger on y-axis (positive); the little finger is on xOy plane (x>0).
    Length normalized by the root bone of the middle finger.
    """
    wrist = keypoint3d[0]
    centered = keypoint3d - wrist
    middle_tip = centered[12]
    assert not (middle_tip[0] == 0.0 and middle_tip[1] == 0.0 and middle_tip[2] == 0.0)
    # rotate around z axis into yOz plane
    alpha = math.pi/2 - math.atan2(middle_tip[1], middle_tip[0])
    c = math.cos(alpha)
    s = math.sin(alpha)
    RzT = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
    rot_z = np.dot(centered, RzT)
    # rotate around x axis into y axis
    middle_tip = rot_z[12]
    beta = - math.atan2(middle_tip[2], middle_tip[1])
    c = math.cos(beta)
    s = math.sin(beta)
    RxT = np.array([[1.0, 0, 0], [0, c, s], [0, -s, c]])
    rot_x = np.dot(rot_z, RxT)
    # rotate around y axis into xOy plane
    pinky_tip = rot_x[20]
    gamma = -math.atan2(pinky_tip[2], pinky_tip[0])
    c = math.cos(gamma)
    s = math.sin(gamma)
    RyT = np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    rot_y = np.dot(rot_x, RyT)
    normalized = rot_y / hand_size(rot_y)
    if flip:
        normalized[:, 2] = -normalized[:, 2]
    return normalized

if __name__ == '__main__':
    # hand_data_a4 with 171026_pose1-3, 171204_pose1-6
    path_to_db = './data/hand_data_a4.json'
    with open(path_to_db) as f:
        json_data_a4 = json.load(f)
    total_data = json_data_a4['training_data'] + json_data_a4['testing_data']
    print('total data num: {}'.format(len(total_data)))

    hands = []
    seqNames = []
    frame_strs = []
    pids = []
    for ihand, hand3d in enumerate(total_data):
        joint3d = np.array(hand3d['hand3d']).reshape(-1, 3)
        canonical = canonical_3d(joint3d, hand3d['lr'])
        hands.append(canonical.reshape(-1))
        seqNames.append(hand3d['seqName'])
        frame_strs.append(hand3d['frame_str'])
        pids.append(hand3d['id'])
    hands = np.array(hands)

    if train:
        kmeans = KMeans(n_clusters=10, random_state=0)
        kmeans.fit(hands)
        hands_cat = kmeans.labels_

        min_sample = 10000000
        for i in range(10):
            print('Cat {}: {}'.format(i, np.sum(hands_cat == i)))
            if np.sum(hands_cat == i) < min_sample:
                min_sample = np.sum(hands_cat == i)
        min_sample = int(0.9*min_sample)
        print ('min_sample: {}'.format(min_sample))

        # print('------------------')
        # for i in range(10):
        #     print('Cat {}: {}'.format(i, np.sum(hands_cat == i)))
        #     center = kmeans.cluster_centers_[i].reshape(-1, 3)
        #     for i in (1, 5, 9, 13, 17):
        #         center[i:i+4] = center[i+3:i-1:-1] # reverse the order of fingers (from palm to tip)
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     plot_hand_3d(center, ax)
        #     plt.show()

        with open('./data/kmeans.pkl', 'wb') as f:
            pickle.dump(kmeans, f)

    if test:
        with open('./data/kmeans.pkl', 'rb') as f:
            kmeans = pickle.load(f)

        np.random.seed(0)
        hands_cat = kmeans.predict(hands)
        chosen_indexes = []
        # choose the hands farthest from the clustering center
        for i in range(10):
            cat_indexes = np.where(hands_cat == i)[0]
            # chosen_indexes.append(np.random.choice(cat_indexes, min_sample))
            dist = nl.norm(hands[hands_cat == i] - kmeans.cluster_centers_[i], axis=1)
            correct_cat_indexes = np.argsort(dist)[:min_sample] # smallest min_sample
            chosen_index = cat_indexes[np.random.choice(correct_cat_indexes, min_sample, replace=False)]
            chosen_indexes.append(chosen_index)
        chosen_indexes = np.array(chosen_indexes)

        # for index in chosen_indexes[9]:
        #     hand = hands[index].reshape(-1, 3)
        #     for i in (1, 5, 9, 13, 17):
        #         hand[i:i+4] = hand[i+3:i-1:-1] # reverse the order of fingers (from palm to tip)
        #     print('{} {} {}'.format(seqNames[index], frame_strs[index], pids[index]))
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     plot_hand_3d(hand, ax)
        #     plt.show()

        chosen_indexes_all = chosen_indexes.reshape(-1).tolist()
        for ihand, hand3d in enumerate(total_data):
            hand3d['resampled'] = int(ihand in chosen_indexes_all)
        with open('./data/hand_data_a4_resampled.json', 'w') as f:
            json.dump(json_data_a4, f)
