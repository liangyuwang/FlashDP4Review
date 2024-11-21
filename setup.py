import os
from setuptools import setup, find_packages

version = '0.0.2'

with open('README.md') as f:
    long_description = f.read()

setup(
    name='flashdp',
    version=version,
    description='Packages of Flash Differential Privacy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires=">=3.11.0",
    include_package_data=True,
    extras_require={
        "test": [
            "tqdm>=4.62.3",
            "pandas>=2.2.0",
            "numpy>=1.26.4",
            "matplotlib>=3.8.3",
        ]
    },
    install_requires=[
        "torch>=2.3.1",
        "triton>=2.3.1",
        "transformers>=4.38.2",
    ],
    test_suite="tests",
    zip_safe=False
)
