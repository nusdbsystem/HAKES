from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="hakes",
    version="0.0.2",
    author="guoyu",
    description="hakes index train lib",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="www.comp.nus.edu.sg/~dbsystem/hakes",
    packages=find_packages(),
    install_requires=[
        "tqdm>=4.66.2",
        "ipywidgets>=8.1.2",
        "scikit-learn>=1.5.2",
        "torch>=1.12.1",
        "numpy>=1.19.5",
    ],
    keywords="DR, IVF, VQ, AKNN",
)
