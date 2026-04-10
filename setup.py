from setuptools import setup, find_packages

setup(
    name='bet',
    version='0.1.0',
    description='Betting analysis and ML pipeline project',
    author='Jim',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.10',
    install_requires=[
        'pyspark==4.1.0',
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'mlflow>=1.0.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.0.0',
    ],
)
