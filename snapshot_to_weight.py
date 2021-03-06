import tensorflow as tf
from utils.general import load_weights
import pickle

# change this
last_cpt = 'snapshots_cpm_rotate_s10_wrist_vgg/model-100000'
# last_cpt = 'snapshots_e2e_RHD_nw/model-100000'
assert last_cpt is not None
weight = load_weights(last_cpt, discard_list=['Adam', 'global_step', 'beta', 'scale'])

# change this
# weight3d = 'snapshots_cpm_rotate_s10_wrist_scale16_dome/model-55000.pickle'
weight3d = 'snapshots_cpm_rotate_s10_wrist_vgg/model-100000.pickle'
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
    
