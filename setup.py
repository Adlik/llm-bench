from setuptools import setup, find_packages

_VERSION = '0.0.0'

_REQUIRED_PACKAGES = [
    'fire==0.5.0',
    'requests==2.31.0',
    'setuptools==45.2.0',
    'numpy==1.24.4',
    'tqdm==4.66.1',
    'pytest==7.4.3',
    'yapf==0.40.2',
    'pandas==2.0.3',
    'pyyaml==6.0.1',
    'jsonlines==4.0.0',
    'datasets==2.15.0',
    'pytest==7.4.3',
    'yapf==0.40.2',
]

_TEST_REQUIRES = [
    'bandit==1.7.4',
    'flake8==4.0.1',
    'pylint==2.6.2'
]

setup(
    name="llm-bench",
    version=_VERSION.replace('-', ''),
    install_requires=_REQUIRED_PACKAGES,
    packages=find_packages(),
    extras_require={'test': _TEST_REQUIRES}
)