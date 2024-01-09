#!/bin/bash

for a in {0..20..1}; do
    A0=$(echo "$a*0.01" | bc -l)
    echo $A0
    sbatch submit.RI 1 ${A0}
    sbatch submit.RI 2 ${A0}
    sbatch submit.RI 3 ${A0}
    sbatch submit.RI 5 ${A0}
    sbatch submit.RI 10 ${A0}
    sbatch submit.RI 100 ${A0}
    sbatch submit.RI 1000 ${A0}
    sbatch submit.RI 10000 ${A0}
    sbatch submit.RI 100000 ${A0}
    sbatch submit.RI 1000000 ${A0}
done

