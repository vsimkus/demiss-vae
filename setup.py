""" Setup script for vgiwae package. """

from setuptools import setup, find_packages

long_description = ('TODO')

setup(
    name="vgiwae",
    author="Vaidotas Simkus",
    description=("Improving Variational Autoencoder Estimation from Incomplete Data with Mixture Variational Families"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/vsimkus/demiss-vae",
    packages=find_packages()
)
