#!/bin/bash
$HOME/.venv/bin/python $HOME/5x10t/dump2positions.py -k 2.80 -s 0 -e 1000 -f dump.npy
$HOME/.venv/bin/python $HOME/5x10t/dump2positions.py -k 3.00 -s 0 -e 1000 -f dump.npy
$HOME/.venv/bin/python $HOME/5x10t/dump2positions.py -k 3.50 -s 0 -e 1000 -f dump.npy
$HOME/.venv/bin/python $HOME/5x10t/dump2positions.py -k 4.00 -s 0 -e 1000 -f dump.npy