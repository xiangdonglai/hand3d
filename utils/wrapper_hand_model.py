# Don't use anaconda for this
import ctypes
import os
from multiprocessing import Process

lib_file = './utils/libPythonWrapper.so'
lib = ctypes.cdll.LoadLibrary(lib_file)

func = lib.fit_hand_and_visualize
func.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_char_p]
func.restype = None

p = ctypes.create_string_buffer("Hi".encode('ascii'), 10)
i = ctypes.c_double(1.0)

# func(i, p)
process = Process(target=func, args=(i, p))
process.start()
process.join()

print("Back here!")