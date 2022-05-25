from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="bayes_tme",
    version="0.1.0",
    description="A reference-free Bayesian method that discovers spatial transcriptional programs in the tissue microenvironment",
    url="https://github.com/jeffquinn-msk/BayesTME",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={  # Optional
        "console_scripts": [
            "grid_search=bayes_tme.grid_search_cfg:main",
            "filter_bleed=bayes_tme.bayestme_filter_bleed:main",
            "deconvolve=bayes_tme.bayestme_deconvolve:main",
            "spatial_expression=bayes_tme.bayestme_spatial_expression:main",
            "prepare_kfold=bayes_tme.bayestme_prepare_kfold:main"
        ],
    },
    python_requires=">=3.7, <4",
    install_requires=[
        "numpy==1.22.3",
        "seaborn==0.11.2",
        "scipy==1.8.1",
        "scikit-image==0.19.2",
        "scikit-learn==1.1.1",
        "pypolyagamma==1.2.3",
        "matplotlib==3.4.3",
        "autograd-minimize==0.2.2",
        "scikit-sparse==0.4.6",
        "torch==1.11.0",
        "torchaudio==0.11.0",
        "torchvision==0.12.0"
    ],
    extras_require={
        "dev": [
            "check-manifest"
        ],
        "test": [
            "pytest",
            "tox"
        ],
    }
)
