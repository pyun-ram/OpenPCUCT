'''
 File Created: Mon Mar 02 2020

'''

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='iou_cpp',
      ext_modules=[cpp_extension.CppExtension('iou_cpp', ['iou.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
setup(name='boxop_cpp',
      ext_modules=[cpp_extension.CppExtension('boxop_cpp', ['boxop.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
setup(name='iou_cuda',
      ext_modules=[cpp_extension.CUDAExtension('iou_cuda', ['iou_cuda.cpp', 'iou_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
setup(name='boxop_cuda',
      ext_modules=[cpp_extension.CUDAExtension('boxop_cuda', ['boxop_cuda.cpp', 'boxop_cuda_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})