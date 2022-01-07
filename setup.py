from setuptools import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="mainProj", sources=["main.pyx"])
setup(ext_modules=cythonize(ext))
