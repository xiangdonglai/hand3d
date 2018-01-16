import os
import pickle as cp
import json
import numpy as np

data_root = '/home/donglaix/Documents/Experiments/STB_train/'

assert(os.path.isdir(data_root))
l = os.listdir(data_root)
l = [_ for _ in l if _.endswith('.json')]
l.sort()

save_data = []
for i, filename in enumerate(l):
    with open(os.path.join(data_root, filename)) as f:
        data = json.load(f)
    if len(data['people']) == 0:
    	continue
    hand2d = np.array(data['people'][0]['hand_left_keypoints']).reshape(-1, 3)[:, :2]
    frame = int(filename.split('_')[0])
    key = './data/stb/stb_train_shuffled.bin:{}'.format(frame)
    save_data.append((key, hand2d))
with open('openpose_stb_train.pkl', 'wb') as f:
    cp.dump(save_data, f)
    
data_root = '/home/donglaix/Documents/Experiments/STB_eval/'

assert(os.path.isdir(data_root))
l = os.listdir(data_root)
l = [_ for _ in l if _.endswith('.json')]
l.sort()

save_data = []
for i, filename in enumerate(l):
    with open(os.path.join(data_root, filename)) as f:
        data = json.load(f)
    if len(data['people']) == 0:
        continue
    hand2d = np.array(data['people'][0]['hand_left_keypoints']).reshape(-1, 3)[:, :2]
    frame = int(filename.split('_')[0])
    key = './data/stb/stb_eval.bin:{}'.format(frame)
    save_data.append((key, hand2d))
with open('openpose_stb_eval.pkl', 'wb') as f:
    cp.dump(save_data, f)
