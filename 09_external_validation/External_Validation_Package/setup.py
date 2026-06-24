# Editable install of the external-validation inference package:
#   pip install -e .
from setuptools import setup, find_packages

setup(
    name="External_Validation_Package",
    version="0.1.0",
    author="External_Validation_Package",
    python_requires=">=3.0",
    packages=find_packages(),
    package_data={
        "External_Validation_Package": ["data/*", "data/**/*"],
    },
    include_package_data=True,
    install_requires=[
        "anndata",
        "numpy",
        "pandas",
        "scipy",
        "xgboost",
        "scikit-learn",
        "gseapy",
        "pynndescent",
        "joblib",
        "scanpy",
    ],
    entry_points={
        "console_scripts": [
            # CLI entry point for the feature-preparation step
            "External_Validation_Package_prepare_anndata = External_Validation_Package.prepare:main",
        ],
    },
)
