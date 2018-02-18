import os
import numpy as np
import json, pickle

root = '/media/posefs0c/panopticdb/a4/'
seqName = '171204_pose5'
frameRange = range(15000, 20000, 5)

calib_file = os.path.join(root, 'annot_calib', seqName, 'calib_00_00.json')
with open(calib_file) as f:
    calib_data = json.load(f)

landmarks = []
img_dirs = []
for i in frameRange:
    img_file = os.path.join(root, 'hdImgs', seqName, '{:08d}'.format(i), '00_00_{:08d}.jpg'.format(i))
    if not os.path.exists(img_file):
        continue
    annot_2d = os.path.join(root, 'annot_hd_2d', seqName, '{:08d}'.format(i), 'Recon2D_00_00_{:08d}.json'.format(i))
    if not os.path.exists(annot_2d):
        continue
    with open(annot_2d) as f:
        data = json.load(f)
    if not 'left_hand' in data[0]:
        continue
    left_hand = data[0]['left_hand']
    if not all(left_hand['insideImg']):
        continue
    landmark = np.array(left_hand['landmarks'])
    img_dirs.append(img_file)
    landmarks.append(landmark)

landmarks = np.array(landmarks)
img_dirs = np.array(img_dirs)

print(len(landmarks), len(img_dirs))

with open('DomeStream.pkl', 'wb') as f:
    pickle.dump((landmarks, img_dirs, calib_data), f)
