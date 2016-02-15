from setuptools import setup

setup(
    name="pymerge",
    version="1.3",
    scripts=['pymerge', 'peaks2hdf', 'evt2hdf'],
    install_requires=['numpy>=1.9',
                      'clint>=0.4',
                      'scipy>=0.16',
                      'h5py>=2.5',
                      'pytpc>=0.7.1'],
    author='Joshua Bradt',
    author_email='bradt@nscl.msu.edu',
    description='A package for merging and processing GRAW data files',
    keywords='graw data merge reduce',
    url='http://github.com/attpc/pymerge'
)
