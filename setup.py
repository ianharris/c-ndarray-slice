from setuptools import setup, Extension
import numpy as np

c_ndarray_slice_module = Extension(
    "slice.c_ndarray_slice",
    sources=["src/module.c", "src/slice.c"],
    include_dirs=[np.get_include()],
    language="c",
)

setup(
    name="slice",
    version="0.1.0",
    description="Python Distribution Utilities",
    author="Ian Harris",
    ext_modules=[c_ndarray_slice_module],
)
