from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='crom_cpu',
    ext_modules=[
        CppExtension(
            name='crom_cpu',
            sources=['crom_linear_cpu.cpp'],
            extra_compile_args=['-fopenmp', '-O3', '-march=native'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
