from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='IEEE Kaggle fraud detection',
    author='LSmith',
    license='MIT',
    install_requires=[
        'pandas==0.25.1',
        'numpy==1.22.0',
        'scikit-learn==0.21.3',
        'catboost==0.16.5',
        'seaborn==0.9.0',
        'hyperopt==0.1.2'
    ]
    ,

)
