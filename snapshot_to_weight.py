import tensorflow as tf
from utils.general import load_weights
import pickle

snapshot_dir = 'snapshots_lifting_proposed_dome'
# last_cpt = tf.train.latest_checkpoint(snapshot_dir)
last_cpt = 'snapshots_lifting_proposed_dome/model-30000'
assert last_cpt is not None
weight = load_weights(last_cpt, discard_list=['Adam', 'global_step', 'beta'])

weight3d = './weights/lifting-proposed-dome.pickle'
with open(weight3d, 'wb') as f:
    pickle.dump(weight, f)

weight_origin = './weights/posenet3d-rhd-stb-slr-finetuned.pickle'
weight_all = './weights/posenet3d-dome.pickle'
with open(weight_origin, 'rb') as f:
    weight_origin = pickle.load(f)
for k, v in weight.items():
    weight_origin[k] = v
with open(weight_all, 'wb') as f:
    pickle.dump(weight_origin, f)
    
