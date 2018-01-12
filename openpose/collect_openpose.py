import os
import pickle as cp
import json
import numpy as np

data_root = '/home/donglaix/Documents/Experiments/detected_hand1/'

assert(os.path.isdir(data_root))
l = os.listdir(data_root)
l = [_ for _ in l if _.endswith('.json')]

save_data = []
for i, filename in enumerate(l):
    with open(os.path.join(data_root, filename)) as f:
        data = json.load(f)
    if len(data['people']) == 0:
    	continue
    hand2d = np.array(data['people'][0]['hand_right_keypoints']).reshape(-1, 3)[:, :2]
    if (hand2d == 0).all():
        continue
    frame_str = filename.split('_')[1]
    save_data.append((frame_str, hand2d))
with open('openpose_data.pkl', 'wb') as f:
    cp.dump(save_data, f)
    
