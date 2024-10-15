#!/bin/bash

for wci in {50..150..5}; do
    for dE in 0.0 0.1 0.2; do
        WC=$(echo "$wci*0.01" | bc -l)
        echo $WC $dE
        sbatch submit.RI 1000 0.1 ${WC} 0.0 ${dE}
        sbatch submit.RI 1000 0.2 ${WC} 0.0 ${dE}
    done
done

