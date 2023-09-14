#!/bin/bash

source $HOME/.bashrc

lscpu
echo "PATH = $PATH"
echo "PYTHONPATH = $PYTHONPATH"

which python
which gcc

cd $HOME/work/git/cpp-numba-blond-bench

python scan.py
