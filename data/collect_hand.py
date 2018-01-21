# This code is written for Python 2.
import os, sys
sys.path.append('../utils/')
import json
import get_file
import numpy as np
import pandas as pd
import cPickle as cp

"""
#################################################################
Panoptic A2
#################################################################
"""

root = '/media/posefs0c/panopticdb/a2/'
panopticmat = os.path.join(root, 'sampleList.mat')


from scipy.io import loadmat
samples = loadmat(panopticmat)['samples'][0, :]
csamples = np.array([(s[0].ravel()[0], s[1].ravel()[0], s[2].ravel(), s[3].ravel()[0], s[4].ravel(), s[5].ravel(), s[6].ravel(), s[7].ravel()) for s in samples],
    dtype=[('seqName', 'O'), ('frame', 'O'), ('camIdxArray', 'O'), ('skelNum', 'O'), ('subjectsWithValidBody', 'O'), ('subjectsWithValidLHand', 'O'), ('subjectsWithValidRHand', 'O'), ('subjectsWithValidFace', 'O')])
df = pd.DataFrame(csamples)


# collect hand data
if os.path.isfile('./hand_data.json'):
    print('Hand file exists.')
else:
    training_data = []
    testing_data = []
    for i, row in df.iterrows():
        if len(row['subjectsWithValidLHand']) == 0 and len(row['subjectsWithValidRHand']) == 0:
            continue
        print('collecting hands {}/{}'.format(i+1, len(df)))
        seqName = row['seqName']
        frame_str = '{:08d}'.format(row['frame'])

        hand3df = get_file.get_annot_file(root, '3dhand', seqName, frame_str)
        with open(hand3df) as f:
            hand3d = json.load(f)

        hands = []
        for person_data in hand3d['people']:
            pid = person_data['id']
            if pid in row['subjectsWithValidLHand']:
                hand_dict = {'seqName': seqName, 'frame_str': frame_str, 'id': pid, 'lr': 0,
                             'hand3d': person_data['left_hand']['landmarks'], 'hand2d': []}
                hands.append(hand_dict)
            if pid in row['subjectsWithValidRHand']:
                hand_dict = {'seqName': seqName, 'frame_str': frame_str, 'id': pid, 'lr': 1,
                             'hand3d': person_data['right_hand']['landmarks'], 'hand2d': []}
                hands.append(hand_dict)

        for camIdx in row['camIdxArray']:
            hand2df = get_file.get_annot_file(root, '2dhand', seqName, frame_str, c=camIdx)
            with open(hand2df) as f:
                hand2d = json.load(f)

            iter_hands = iter(hands)
            for person_data in hand2d:
                pid = person_data['id']
                if pid in row['subjectsWithValidLHand']:
                    hand_dict = iter_hands.next()
                    assert pid == hand_dict['id']
                    if not any(person_data['left_hand']['self_occluded'][1:]) and not person_data['left_hand']['overlap'] and all(person_data['left_hand']['insideImg']):
                        # valid hand from this view point
                        hand_dict['hand2d'].append(int(camIdx))
                if pid in row['subjectsWithValidRHand']:
                    hand_dict = iter_hands.next()
                    assert pid == hand_dict['id']
                    if not any(person_data['right_hand']['self_occluded'][1:]) and not person_data['right_hand']['overlap'] and all(person_data['right_hand']['insideImg']):
                        # valid hand from this view point
                        hand_dict['hand2d'].append(int(camIdx))

            assert len(list(iter_hands)) == 0 # check the number of hands

        for hand in hands:
            if len(hand['hand2d']) > 0:
                if seqName.startswith('170407'):
                    testing_data.append(hand)
                else:
                    training_data.append(hand)

    with open('./hand_data.json', 'w') as f:
        json.dump({'training_data': training_data, 'testing_data': testing_data}, f)

# collect camera calibration data
if os.path.isfile('./camera_data.pkl'):
    print('Camere file exists.')
else:
    seqs = df['seqName'].unique()
    calib_dict = {}
    for seqName in seqs:
        cam_dict = {}
        for camIdx in xrange(31):
            calib_file = get_file.get_annot_file(root, 'calib', seqName, c=camIdx)
            calib = get_file.load_calib_file(calib_file)
            cam_dict[camIdx] = calib
        calib_dict[seqName] = cam_dict
    with open('./camera_data.pkl', 'wb') as f:
        cp.dump(calib_dict, f)

"""
#################################################################
Panoptic A4
#################################################################
"""

root = '/media/posefs0c/panopticdb/a4/'
sample_list = os.path.join(root, 'sample_list.pkl')

with open(sample_list, 'rb') as f:
    df = cp.load(f)

# collect hand data
if os.path.isfile('./hand_data_a4.json'):
    print('Hand file exists.')
else:
    training_data = []
    testing_data = []
    for seqName, seq_samples in df.iteritems():
        i = 0
        for hvframe, frame_dict in seq_samples.iteritems():
            i += 1
            hv, frame_str = hvframe
            print('collecting hands: {} {}/{}'.format(seqName, i, len(seq_samples)))
            hand3df = os.path.join(root, 'annot_{}_3d'.format(hv), seqName, 'Recon3D_{0}{1}.json'.format(hv, frame_str))
            with open(hand3df) as f:
                hand3d = json.load(f)

            hands = []
            for person_data in hand3d:
                pid = person_data['id']
                if 'subjectsWithValidLHand' in frame_dict and pid in frame_dict['subjectsWithValidLHand']:
                    hand_dict = {'seqName': seqName, 'frame_str': frame_str, 'id': pid, 'lr': 0,
                                 'hand3d': person_data['left_hand']['landmarks'], 'hand2d': []}
                    hands.append(hand_dict)
                if 'subjectsWithValidRHand' in frame_dict and pid in frame_dict['subjectsWithValidRHand']:
                    hand_dict = {'seqName': seqName, 'frame_str': frame_str, 'id': pid, 'lr': 1,
                                 'hand3d': person_data['right_hand']['landmarks'], 'hand2d': []}
                    hands.append(hand_dict)

            for panelIdx, camIdx in frame_dict['camIdxArray']:
                hand2df = os.path.join(root, 'annot_{}_2d'.format(hv), seqName, frame_str, 'Recon2D_00_{0:02d}_{1}.json'.format(camIdx, frame_str))
                with open(hand2df) as f:
                    hand2d = json.load(f)

                iter_hands = iter(hands)
                for person_data in hand2d:
                    pid = person_data['id']
                    if 'subjectsWithValidLHand' in frame_dict and pid in frame_dict['subjectsWithValidLHand']:
                        hand_dict = iter_hands.next()
                        assert pid == hand_dict['id']
                        if not any(person_data['left_hand']['self_occluded'][1:]) and not person_data['left_hand']['overlap'] and all(person_data['left_hand']['insideImg']):
                            # valid hand from this view point
                            hand_dict['hand2d'].append(int(camIdx))
                    if 'subjectsWithValidRHand' in frame_dict and pid in frame_dict['subjectsWithValidRHand']:
                        hand_dict = iter_hands.next()
                        assert pid == hand_dict['id']
                        if not any(person_data['right_hand']['self_occluded'][1:]) and not person_data['right_hand']['overlap'] and all(person_data['right_hand']['insideImg']):
                            # valid hand from this view point
                            hand_dict['hand2d'].append(int(camIdx))

                assert len(list(iter_hands)) == 0 # check the number of hands

            for hand in hands:
                if len(hand['hand2d']) > 0:
                    if seqName == '171204_pose5':
                        continue
                    if seqName == '171204_pose4':
                        testing_data.append(hand)
                    else:
                        training_data.append(hand)

    with open('./hand_data_a4.json', 'w') as f:
        json.dump({'training_data': training_data, 'testing_data': testing_data}, f)

# collect camera calibration data
if os.path.isfile('./camera_data_a4.pkl'):
    print('Camere file exists.')
else:
    seqs = df.keys()
    calib_dict = {}
    for seqName in seqs:
        cam_dict = {}
        for camIdx in xrange(31):
            calib_file = get_file.get_annot_file(root, 'calib', seqName, c=camIdx)
            calib = get_file.load_calib_file(calib_file)
            cam_dict[camIdx] = calib
        calib_dict[seqName] = cam_dict
    with open('./camera_data_a4.pkl', 'wb') as f:
        cp.dump(calib_dict, f)