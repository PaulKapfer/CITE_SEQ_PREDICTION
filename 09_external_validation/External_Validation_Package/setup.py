from setuptools import find_packages, setup

setup(
    name="citepredict",
    version="0.2.0",
    description="Predict surface-protein (ADT) abundance from scRNA-seq with a CITE-seq-trained XGBoost model",
    python_requires=">=3.10",
    packages=find_packages(),
    package_data={"citepredict": ["data/*", "data/**/*"]},
    include_package_data=True,
    install_requires=[
        "anndata",
        "numpy",
        "pandas",
        "scipy",
        "xgboost",
        "joblib",
    ],
    entry_points={
        "console_scripts": [
            "citepredict-prepare = citepredict.prepare:main",
        ],
    },
)
