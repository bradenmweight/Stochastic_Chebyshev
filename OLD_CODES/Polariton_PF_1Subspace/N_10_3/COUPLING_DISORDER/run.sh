#!/bin/bash

for wci in {50..150..5}; do
    WC=$(echo "$wci*0.01" | bc -l)
    echo $WC
    sbatch submit.RI 1000 0.1 ${WC} 0.0
    sbatch submit.RI 1000 0.2 ${WC} 0.0
done

