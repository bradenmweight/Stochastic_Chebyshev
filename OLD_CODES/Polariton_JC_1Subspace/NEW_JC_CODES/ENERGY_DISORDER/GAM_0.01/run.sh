#!/bin/bash

for A0 in 0.05; do
    for wci in 100; do
        WC=$(echo "$wci*0.01" | bc -l)
        for CL in 0.0; do
        #for CL in 0.0 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24; do
            #for SIGE in 0.0 0.02 0.04 0.06 0.08 0.10; do
            for SIGE in 0.0; do
            #for SIGE in 0.0 0.04 0.08 0.12; do
                for SIGG in 0.0; do
                    echo ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                    sbatch submit.RI 100 ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                    #sbatch submit.RI 10000 ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                    #sbatch submit.RI 100000 ${A0} ${WC} ${CL} ${SIGE} ${SIGG}
                    sleep 2
                done
            done
        done
    done
done

