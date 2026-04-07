#!/bin/bash
#BSUB -q long
#BSUB -cwd /home/labs/lancet/Collaboration/liors/SourceCode/Doron_Lancet_GARD_Next_generation/Lior_scripts/Testrun
#BSUB -o Testrun.o%J
#BSUB -e Testrun.e%J
#BSUB -J Testrun
#BSUB -R "select[mem>2GB]"
#BSUB -n 1
NAME=SimpleGard10Simulation
date
hostname
free -g
module load matlab/R2018a
pwd
cat $NAME.m
matlab -singleCompThread -nodisplay -nojvm -r $NAME
date

