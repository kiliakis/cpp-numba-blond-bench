#!/bin/bash

source $HOME/.bashrc

lscpu
echo "PATH = $PATH"
echo "PYTHONPATH = $PYTHONPATH"

which python
which gcc

cd $HOME/work/git/cpp-numba-blond-bench

# python driver.py -suite numba_par cpp -b histo_v5 lik lik_v3 kick drift -i 100 --nrf 1 -r 10 -output results-2023-09-15-kick-drift-lik-histo-v2.csv -t 1 -p 1000000 8000000 -s 128 1024
# python driver.py -suite numba_par cpp -b histo_v5 lik lik_v3 kick drift -i 100 --nrf 1 -r 10 -output results-2023-09-15-kick-drift-lik-histo-v2.csv -t 8 -p 1000000 8000000 -s 128 1024


python driver.py -suite numba_par cpp -b kick -i 100 --nrf 1 -r 10 -output results-2023-09-15-kick.csv -t 1 -p 1000000 8000000 -s 128 1024
python driver.py -suite numba_par cpp -b kick -i 100 --nrf 1 -r 10 -output results-2023-09-15-kick.csv -t 8 -p 1000000 8000000 -s 128 1024
