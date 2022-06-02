"""Package setup."""

import setuptools

__version__ = None
exec(open("flim/_version.py").read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FLIM",
    version=__version__,
    author="Italos Estilon de Souza",
    author_email="italosestilon@gmail.com",
    packages=setuptools.find_packages(),
    description="A package to build deep learning models from images markers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LIDS-UNICAMP/FLIM",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Education :: Developers",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: \
            Artificial Intelligence :: Image Recognition",
        "Topic :: Software Development :: Libraries",
    ],
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.2",
        "numpy>=1.20",
        "scikit-image>=0.19.2",
        "scikit-learn>=1.0.0",
        "setuptools>=62.3.2",
        "numba>=0.55.2",
        "termcolor>=1.1.0",
        "nibabel>=3.2.2",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "train=flim.tools.training_tool:main",
            "validate=flim.tools.validating_tool:main",
            "annotation=flim.tools.annotation_tool:main",
        ]
    },
)
