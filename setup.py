from setuptools import find_packages, setup

setup(
    name='mvaelib',
    packages=find_packages(include=['mvaelib']),
    version='0.0.3',
    description='Synthetic multimodal profiles generator',
    author='Anthony Lysenko, Shikov Egor, Deeva Irina',
    license='MIT',
    install_requires=['pandas', 'numpy', 'torch', 'scikit-learn', 'joblib'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests'
)
