#!/bin/bash

for wci in {50..150..5}; do
    WC=$(echo "$wci*0.01" | bc -l)
    echo $WC
    sbatch submit.RI 10000 0.3 ${WC}
done

