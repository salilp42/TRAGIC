from setuptools import setup, find_packages

setup(
    name="tragic",
    version="0.1.0",
    description="Time seRies Analysis using Graph attentIon networks Classifier",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "tslearn>=0.5.2",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "jupyter>=1.0.0"
    ],
    python_requires=">=3.7",
)
