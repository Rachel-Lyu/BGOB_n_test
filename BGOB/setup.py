from setuptools import setup, find_packages

setup(
    name="BGOB",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "tqdm",
        "numpy",
        "pandas",
        "scikit-learn"
    ],
)