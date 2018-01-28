import tensorflow as tf
from utils.general import load_weights
import pickle

# change this
last_cpt = 'snapshots_cpm_rotate_s10_vgg/model-85000'
assert last_cpt is not None
weight = load_weights(last_cpt, discard_list=['Adam', 'global_step', 'beta'])

# change this
weight3d = './weights/cpm_tf.pickle'
with open(weight3d, 'wb') as f:
    pickle.dump(weight, f, protocol=2)

# weight_origin = './weights/posenet3d-rhd-stb-slr-finetuned.pickle'
# # change this
# weight_all = './weights/posenet3d-dome-a4-resampled.pickle'
# with open(weight_origin, 'rb') as f:
#     weight_origin = pickle.load(f)
# for k, v in weight.items():
#     weight_origin[k] = v
# with open(weight_all, 'wb') as f:
#     pickle.dump(weight_origin, f)
    
