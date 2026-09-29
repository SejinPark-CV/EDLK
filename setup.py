from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="difflogic",
    version="0.1.0",

    author="Felix Petersen",
    author_email="ads0600@felix-petersen.de",

    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/Felix-Petersen/difflogic",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    package_dir={
        "difflogic": "difflogic",
    },

    packages=[
        "difflogic",
    ],

    # EDLK only requires the original difflogic CUDA extension.
    # The IWP extension is intentionally not built.
    ext_modules=[
        CUDAExtension(
            name="difflogic_cuda",
            sources=[
                "difflogic/cuda/difflogic.cpp",
                "difflogic/cuda/difflogic_kernel.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O2",
                ],
                "nvcc": [
                    "-O2",
                    "-lineinfo",
                ],
            },
        ),
    ],

    cmdclass={
        "build_ext": BuildExtension,
    },

    python_requires=">=3.8",

    install_requires=[
        "torch",
        "numpy",
    ],
)