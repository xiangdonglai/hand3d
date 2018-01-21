import os, json
import numpy as np

def get_annot_file(root, annot_type, seq, frame_str='00000000', p=0, c=0):
    assert annot_type in ['2dhand', '3dhand', '3dskeleton', '2dskeleton', '3dface', '2dface', 'calib']
    assert type(frame_str) == str
    annot_dir = os.path.join(root, 'annot_' + annot_type, seq)
    if '2d' in annot_type:
        annot_dir = os.path.join(annot_dir, frame_str)

    if annot_type == '2dhand':
        file = os.path.join(annot_dir, 'hand2D_{:02d}_{:02d}_{:}.json'.format(p, c, frame_str))
    elif annot_type == '3dhand':
        file = os.path.join(annot_dir, 'handRecon3D_hd{}.json'.format(frame_str))
    elif annot_type == '2dskeleton':
        file = os.path.join(annot_dir, 'body2DScene_{:02d}_{:02d}_{:}.json'.format(p, c, frame_str))
    elif annot_type == '3dskeleton':
        file = os.path.join(annot_dir, 'body3DScene_{}.json'.format(frame_str))
    elif annot_type == '2dface':
        file = os.path.join(annot_dir, 'face2D_{:02d}_{:02d}_{:}.json'.format(p, c, frame_str))
    elif annot_type == '3dface':
        file = os.path.join(annot_dir, 'faceRecon3D_hd{}.json'.format(frame_str))
    elif annot_type == 'calib':
        file = os.path.join(annot_dir, 'calib_{:02d}_{:02d}.json'.format(p, c))
    else:
        raise NotImplementedError

    return file

def get_img_file(root, seq, frame_str, p=0, c=0):
    return os.path.join(root, 'imgs', seq, frame_str, '{:02d}_{:02d}_{:}.jpg'.format(p, c, frame_str))

def get_crop_dir(root, annot_type, seq):
    return os.path.join(root, 'cropped_' + annot_type, seq)

def load_calib_file(calib_file):
    assert os.path.isfile(calib_file)
    with open(calib_file) as f:
        calib = json.load(f)
    for key in calib:
        if type(calib[key]) == list:
            calib[key] = np.array(calib[key])
    return calib

def get_annot_dir(root, annot_type, seq, frame_str='00000000'):
    assert annot_type in ['2dhand', '3dhand', '2dskeleton', '3dskeleton', 'calib', '2dface']
    assert type(frame_str) == str
    annot_dir = os.path.join(root, 'annot_' + annot_type, seq)
    if '2d' in annot_type:
        annot_dir = os.path.join(annot_dir, frame_str)
    return annot_dir