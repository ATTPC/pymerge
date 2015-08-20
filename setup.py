from setuptools import setup

setup(
    name = "pymerge",
    version = "1.1",
    scripts = ['pymerge'],

    install_requires = ['numpy>=1.8',
                        'clint>=0.4',
                        'scipy>=0.16',
                        'pytpc'],

    author = 'Joshua Bradt',
    author_email = 'bradt@nscl.msu.edu',
    description = 'A package for merging GRAW data files',
    keywords = 'graw data merge',
    url = 'http://github.com/attpc/pymerge'
)
