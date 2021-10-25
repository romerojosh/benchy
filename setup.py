from setuptools import setup, find_packages

setup(
    name='benchy',
    packages=find_packages(exclude=('benchy.egg-info',
                                    'build',
                                    'dist',)),
    version='0.1',
    description='Benchmarking Dataloader Wrapper for DL',
)
