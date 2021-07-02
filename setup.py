from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lambertw',
    ext_modules=[
        CUDAExtension('lambertw', [
            'kernels/lambert.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='spiketime',
    ext_modules=[
        CUDAExtension('spiketime', [
            'kernels/spiketime.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

'''
Run the command below to build cuda extension locally
python setup.py build_ext --inplace
'''