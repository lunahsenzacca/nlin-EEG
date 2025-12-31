from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy

extensions = [
    Extension('c_core', ['./cython_modules/c_core.pyx'],
        include_dirs=[numpy.get_include()])
]

setup(
    ext_modules = cythonize(extensions)
)