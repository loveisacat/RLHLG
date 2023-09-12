#!/bin/bash
 
#PBS -l ncpus=16
#PBS -l mem=5GB
#PBS -l jobfs=1GB
#PBS -q normal
#PBS -P kq93 
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/bc3607+scratch/bc3607
#PBS -l wd
  
module load python3/3.10.0
python3 -m scripts.train --algo ppo --env MiniGrid-LavaCrossingS9N1-v0 --model crossing_task_A_0 --save-interval 100 --frames 2000000  --seed 9 > result/crossing_task_A
python3 -m scripts.train --algo ppo --env MiniGrid-LavaCrossingS9N1-v0 --model crossing_task_A_1 --save-interval 100 --frames 2000000  --seed 10 > result/crossing_task_A

