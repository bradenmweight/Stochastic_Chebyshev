#!/bin/bash

#for A0 in 0.0 0.01 0.02 0.03 0.04 0.05; do
for A0 in 0.0 0.01 0.02 0.03 0.04 0.05; do
    for wci in 100; do
        WC=$(echo "$wci*0.01" | bc -l)
        for CL in 0.0 0.01 0.02 0.03 0.04 0.05; do
            for SIGE in 0.0 0.01 0.02 0.03 0.04 0.05; do
                for SIGG in 0.0; do
                    echo ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                    sbatch submit.RI 10 ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                done
            done
        done
    done
done

