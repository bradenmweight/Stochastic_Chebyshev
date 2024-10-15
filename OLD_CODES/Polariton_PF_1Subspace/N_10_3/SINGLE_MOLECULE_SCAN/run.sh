#!/bin/bash


for wci in {50..150..10}; do
    for ERxNi in {500..1500..100}; do
        WC=$(echo "$wci*0.01" | bc -l)
        ERxN=$(echo "$ERxNi*0.001" | bc -l)
        echo $WC $ERxN
        sbatch submit.RI 1000 0.1 ${WC} ${ERxN}
        sbatch submit.RI 1000 0.2 ${WC} ${ERxN}
    done
done

