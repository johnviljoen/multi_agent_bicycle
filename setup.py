from setuptools import setup, find_packages

setup(
    name='multi_agent_bicycle',
    version='0.0.1',
    author='John Viljoen',
    author_email='johnviljoen2@gmail.com',
    install_requires=[
        'matplotlib',   # plotting...
        'tqdm',         # just for pretty loops in a couple places
        'numpy',
        'jax[cuda12]'
    ],
    packages=find_packages(include=[]),
)

