from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(ext_modules=cythonize('hdial_libc.pyx'), include_dirs=[np.get_include()], requires=['cv2'])
