#!/bin/bash

for nmol in 1 11 101 501 1001; do
    for nmode in 1 11 101 501 1001; do
        echo "nmol = $nmol, nmode = $nmode"
        sbatch submit.RI $nmol $nmode
    done
done