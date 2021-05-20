import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "DeepAnalogs", "version.py")) as fp:
    exec(fp.read())

setuptools.setup(
    name="DeepAnalogs",
    version=__version__,
    author="Weiming Hu",
    author_email="huweiming950714@gmail.com",
    description="Training a deep network for weather analogs",
    url="https://github.com/Weiming-Hu/DeepAnalogs",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "deep_analogs_train = DeepAnalogs.train:main",]},
    python_requires=">=3",
    license='LICENSE',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'prettytable',
        'pyyaml',
        'netCDF4',
        'tqdm',
        'bottleneck',
        'numpy',
        'torch',
    ],
)

