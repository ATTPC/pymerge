# PyMerge

This is a Python-based merger for the GRAW files produced by the AT-TPC. (The Prototype AT-TPC is not supported currently.)
This is a (potential) replacement for the C++ merger at https://github.com/attpc/get-manip.

## Dependencies

The program requires these packages to work:
- pytpc (see https://github.com/attpc/pytpc)
- clint
- numpy
- scipy

## Installation

It is best to install numpy and scipy using conda. This will also be required to install pytpc, so see the directions
in the README for pytpc at https://github.com/attpc/pytpc.

After installing those packages, install this program using
```
python setup.py install
```

## Contact

Josh Bradt, bradt@nscl.msu.edu
