import tensorflow as tf
import numpy as np

class MultiDataset(object):
    #  A class to combine multi dataset input
    def __init__(self, db_list):
        assert type(db_list) == list and len(db_list) >= 1
        self.db_list = db_list

    def get(self, read_image=False, extra=False):
        data_list = []
        intersection_name = set()
        union_name = set()
        for i, db in enumerate(self.db_list):
            if type(db.path_to_db) == str and db.path_to_db.endswith('.bin'):
                # BinaryDbReader & BinaryDbReaderSTB: read image from .bin file
                data = db.get(extra)
            else:
                data = db.get(read_image=read_image, extra=extra)

            names = set(data.keys())
            data_list.append(data)

            if i == 0:
                intersection_name |= names
            else:
                intersection_name &= names
            union_name |= names

        deleted_names = union_name - intersection_name
        print('Warning: removing uncommon data entries: {}'.format(list(deleted_names)))

        data = {}
        intersection_name = intersection_name
        for name in intersection_name:
            data[name] = tf.concat([d[name] for d in data_list], axis=0)

        return data


if __name__ == '__main__':
    from data.DomeReader import DomeReader
    from data.BinaryDbReader import BinaryDbReader
    from data.TsimonDBReader import TsimonDBReader
    # d1 = BinaryDbReader(mode='training', shuffle=False, hand_crop=True, use_wrist_coord=False)
    d2 = DomeReader(mode='training', shuffle=False, hand_crop=True, use_wrist_coord=False, a4=False, crop_size=368, batch_size=8)
    d3 = TsimonDBReader(mode='training', shuffle=False, hand_crop=True, use_wrist_coord=False, crop_size=368, batch_size=4)

    d = MultiDataset([d2, d3])
    data = d.get(read_image=True, extra=True)


