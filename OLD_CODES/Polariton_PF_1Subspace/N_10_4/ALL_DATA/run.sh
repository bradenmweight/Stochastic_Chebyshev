#!/bin/bash

for A0 in 0.1; do
    for wci in {50..150..5}; do
        WC=$(echo "$wci*0.01" | bc -l)
        for CL in 0.0 0.1 0.2 0.3 0.4 0.5; do
            for SIGE in 0.0 0.1 0.2; do
                for SIGG in 0.0 1.0; do
                    echo ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                    sbatch submit.RI 10000 ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                done
            done
        done
    done
done

