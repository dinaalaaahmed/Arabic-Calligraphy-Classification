from setuptools import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext = Extension(name="hello", sources=["hello2.pyx"])
setup(ext_modules=cythonize(ext))
