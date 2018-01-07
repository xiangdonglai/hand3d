import os, cv2
import tensorflow as tf
import numpy as np

from data.BinaryDbReaderSTB import BinaryDbReaderSTB

# root_dir = '/home/donglaix/Documents/Experiments/STB_eval/'
root_dir = '/home/donglaix/Documents/Experiments/STB_train/'

dataset = BinaryDbReaderSTB(mode='training', shuffle=False, use_wrist_coord=False)
data = dataset.get()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)

for i in range(dataset.num_samples):
    key, image = sess.run([data['key'], data['image']])
    image = np.squeeze((image+0.5)*255).astype(np.uint8)
    no = int(key[0].decode().split(':')[1])
    imname = '{0:05d}.jpg'.format(no)

    cv2.imwrite(os.path.join(root_dir, imname), image[:, :, ::-1])