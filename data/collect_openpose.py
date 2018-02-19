import os
import numpy as np
import json, pickle

root = '/home/donglaix/Documents/Experiments/'

calib_file = os.path.join(root, 'calibration', 'calib.json')
with open(calib_file) as f:
    calib_data = json.load(f)

frameRange = range(0, 392)

landmarks = []
img_dirs = []
for i in frameRange:
    img_file = os.path.join(root, 'openpose_image_test1', 'handTest_{:012d}_rendered.png'.format(i))
    assert os.path.exists(img_file)
    annot_2d = os.path.join(root, 'detected_hand_test1', 'handTest_{:012d}_keypoints.json'.format(i))
    assert os.path.exists(annot_2d)
    with open(annot_2d) as f:
        data = json.load(f)
    joint2d = np.array(data["people"][0]["hand_left_keypoints"]).reshape(-1, 3)[:, :2]
    if np.array(data["people"][0]["hand_left_keypoints"]).reshape(-1, 3)[:, 2].any():
        landmarks.append(joint2d)
    else:
        landmarks.append(landmarks[-1])
    img_dirs.append(img_file)

landmarks = np.array(landmarks)
img_dirs = np.array(img_dirs)

print(len(landmarks), len(img_dirs))

with open('DomeStreamTest1.pkl', 'wb') as f:
    pickle.dump((landmarks, img_dirs, calib_data), f)
