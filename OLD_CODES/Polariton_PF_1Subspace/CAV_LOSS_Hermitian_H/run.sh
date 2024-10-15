#!/bin/bash

for wci in {50..150..5}; do
    WC=$(echo "$wci*0.01" | bc -l)
    for CL in 0.0 0.1 0.2 0.3 0.4 0.5; do
        echo $WC $CL
        sbatch submit.RI 10000 0.1 ${WC} ${CL}
        sbatch submit.RI 10000 0.2 ${WC} ${CL}
    done
done

