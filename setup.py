from setuptools import setup

setup(
    name='FLIM',
    version='0.1',
    author='Italos Estilon de Souza',
    author_email='italosestilon@gmail.com',
    packages=setuptools.find_packages(),
    description='A package to build deep learning models from images markers.',
    install_requires=[
        "torch",
        "numpy",
        "scikit-image",
        "sklearn",
        "setuptools"
    ]
)