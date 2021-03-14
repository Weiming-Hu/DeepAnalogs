import setuptools

setuptools.setup(
    name="DeepAnalogs",
    version="0.0.0",
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
        'configargparse',
        'netCDF4',
        'tqdm',
        'bottleneck',
        'numpy',
        'torch',
    ],
)

