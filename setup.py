from setuptools import setup, find_packages
import limo

setup(name='limo',
      version=limo.__version__,
      description='Generalized linear models',
      author='Niru Maheshwaranathan',
      author_email='nirum@stanford.edu',
      url='https://github.com/nirum/limo.git',
      requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
          The limo package is a set of tools for creating and fitting generalized linear
          models (GLMs), specifically tuned for sensory neuroscience data and applications.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=find_packages(),
      license='LICENSE.md'
      )
