# Don't use anaconda for this
import ctypes
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class wrapper_hand_model(object):
    def __init__(self, lib_file='./utils/libPythonWrapper.so', model_file='./utils/hand2_l_all_uv.json'):
        self.lib = ctypes.cdll.LoadLibrary(lib_file)
        self.func = self.lib.fit_hand_and_visualize
        self.func.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.POINTER(ctypes.c_ubyte)]
        self.func.restype = None

        self.cmodel_file = ctypes.create_string_buffer(model_file.encode('ascii'))
        self.ctarget_array = (ctypes.c_double*63)()
        self.cret_bytes = (ctypes.c_ubyte*(600*600*3))()

    def fit_render(self, joint3d):
        assert joint3d.shape == (21, 3)
        self.ctarget_array[:] = joint3d.reshape(-1).tolist()
        self.func(self.ctarget_array, self.cmodel_file, self.cret_bytes)
        img = bytes(self.cret_bytes)
        img = Image.frombytes("RGB", (600, 600), img)
        img = ImageOps.flip(img)
        return img


if __name__ == '__main__':
    import numpy as np
    wrapper = wrapper_hand_model()
    joint3d = np.array([-33.3889, -173.355, -36.0744, -35.0518, -173.959, -37.7108, -36.5972, -176.126, -40.7544, -37.4367, -178.032, -43.6272, -38.7743, -178.843, -45.5877, -36.4731, -180.718, -38.2183, -37.0009, -181.596, -42.4443, -37.4651, -181.437, -45.0006, -37.7732, -181.458, -47.0573, -34.2598, -180.606, -38.3926, -35.2143, -180.671, -43.2699, -36.3031, -179.876, -45.6931, -37.1902, -179.438, -47.745, -32.0926, -179.69, -38.4972, -33.7518, -179.847, -42.8798, -34.9357, -179.212, -45.3947, -35.7699, -178.853, -47.3468, -30.3247, -178.334, -39.2571, -31.8778, -178.837, -42.4667, -33.003, -178.501, -44.2697, -33.8762, -178.325, -45.8248]).reshape(-1, 3)
    img = wrapper.fit_render(joint3d)
    plt.imshow(img)
    plt.show()
