#!/bin/bash

#for A0 in 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.15 0.20; do
for A0 in 0.01; do
    #for wci in {50..150..5}; do
    for wci in 100; do
        WC=$(echo "$wci*0.01" | bc -l)
        #for CL in 0.0 0.1 0.2 0.3 0.4 0.5; do
        for CL in 0.0; do
            for SIGE in 0.0 0.005 0.01 0.015 0.02 0.05; do
                for SIGG in 0.0; do
                    echo ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                    #sbatch submit.RI 1000 ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                    sbatch submit.RI 100000 ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                done
            done
        done
    done
done

