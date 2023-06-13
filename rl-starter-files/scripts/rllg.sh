#!/bin/bash
 
#PBS -l ncpus=16
#PBS -l mem=5GB
#PBS -l jobfs=1GB
#PBS -q express
#PBS -P kq93 
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/bc3607+scratch/bc3607
#PBS -l wd
  
module load python3/3.10.0
#python3 -m scripts.train --algo ppo --env MiniGrid-LavaCrossingS9N2-v0 --model distshift_B_0612 --save-interval 100 --frames 40000000 --seed 9 > result/result_B_rllg_0612
python3 -m scripts.train --algo ppo --env MiniGrid-LavaCrossingS9N3-v0 --model distshift_C_0612 --save-interval 100 --frames 30000000 --seed 9 > result/result_C_rllg_0612
