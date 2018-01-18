import tensorflow as tf
from utils.general import load_weights
import pickle

# change this
last_cpt = 'snapshots_joint_domeaug-a4/model-80000'
assert last_cpt is not None
weight = load_weights(last_cpt, discard_list=['Adam', 'global_step', 'beta'])

# change this
weight3d = './weights/lifting-domeaug-a4-jft.pickle'
with open(weight3d, 'wb') as f:
    pickle.dump(weight, f)

weight_origin = './weights/posenet3d-rhd-stb-slr-finetuned.pickle'
# change this
weight_all = './weights/posenet3d-domeaug-a4-jft.pickle'
with open(weight_origin, 'rb') as f:
    weight_origin = pickle.load(f)
for k, v in weight.items():
    weight_origin[k] = v
with open(weight_all, 'wb') as f:
    pickle.dump(weight_origin, f)
    
