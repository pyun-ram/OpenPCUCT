import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources, **kwargs):
    root_dir = os.path.abspath(__file__)
    root_dir = os.path.dirname(root_dir)
    include_dirs = kwargs.get("include_dirs", [])
    include_dirs = [os.path.join(root_dir, *module.split('.'), inc) for inc in include_dirs]
    kwargs.pop("include_dirs")
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        include_dirs=include_dirs,
        **kwargs
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.1+%s' % get_git_commit_number()

    setup(
        name='pcuct',
        version=version,
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='jiou_3d_cpp',
                module='pcuct.ops.jiou',
                sources=[
                    'src/jiou_3d.cpp',
                ],
                include_dirs=["include/eigen/"],
                extra_compile_args=['-fopenmp']
            ),
            make_cuda_ext(
                name='jiou_bev_cpp',
                module='pcuct.ops.jiou',
                sources=[
                    'src/jiou_bev.cpp',
                ],
                include_dirs=["include/eigen/"],
                extra_compile_args=['-fopenmp']
            ),
        ],
    )
