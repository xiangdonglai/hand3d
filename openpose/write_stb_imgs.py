import os, cv2
import tensorflow as tf
import numpy as np
import pdb

from data.BinaryDbReaderSTB import BinaryDbReaderSTB
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import single_obj_scoremap, calc_center_bb

root_dir = '/home/donglaix/Documents/Experiments/STB_eval/'
bb_dir = '/home/donglaix/Documents/Experiments/STB_eval_BB/'
# root_dir = '/home/donglaix/Documents/Experiments/STB_train/'
# bb_dir = '/home/donglaix/Documents/Experiments/STB_train_BB/'

dataset = BinaryDbReaderSTB(mode='evaluation', shuffle=False, use_wrist_coord=False)
# dataset = BinaryDbReaderSTB(mode='training', shuffle=False, use_wrist_coord=False)
data = dataset.get()
image_scaled = tf.image.resize_images(data['image'], (240, 320))
net = ColorHandPose3DNetwork()
hand_scoremap = net.inference_detection(image_scaled)
hand_scoremap = hand_scoremap[-1]
hand_mask = single_obj_scoremap(hand_scoremap)
center, _, crop_size_best = calc_center_bb(hand_mask)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

net.init(sess, weight_files=['./weights/handsegnet-rhd.pickle'])

for i in range(dataset.num_samples):
    key, image, center_v, crop_size = sess.run([data['key'], data['image'], center, crop_size_best])
    image = np.squeeze((image+0.5)*255).astype(np.uint8)
    no = int(key[0].decode().split(':')[1])
    imname = '{0:05d}.jpg'.format(no)

    center_v *= 2
    crop_size *= (2 * 1.8)

    # cv2.imwrite(os.path.join(root_dir, imname), image[:, :, ::-1])
    x = int(center_v[0, 1] - crop_size[0, 0]/2)
    y = int(center_v[0, 0] - crop_size[0, 0]/2)
    w = h = int(crop_size[0, 0])
    with open(os.path.join(bb_dir, '{0:05d}_l.txt'.format(no)), 'w') as f:
    	f.write('{} {} {} {}'.format(x, y, w, h))
    print(no)

