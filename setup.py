"""Package setup."""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='FLIM',
    version='0.1',
    author='Italos Estilon de Souza',
    author_email='italosestilon@gmail.com',
    packages=setuptools.find_packages(),
    description='A package to build deep learning models from images markers.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/italosestilon/FLIM",
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
        "Topic :: Software Development :: Libraries"
    ],
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "scikit-image",
        "sklearn",
        "setuptools",
        "numba",
        "termcolor"
    ],
    python_requires='>=3.7',
    entry_points={'console_scripts': ['train=flim.tools.training_tool:main',
                                      'validate=flim.tools.validating_tool:main'
                                     ]}
)
