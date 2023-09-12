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
python3 -m scripts.train --algo ppo --env MiniGrid-LavaCrossingS9N2-v0 --model crossing_task_B_0 --save-interval 100 --frames 5000000  --seed 9 > result/crossing_task_A
python3 -m scripts.train --algo ppo --env MiniGrid-LavaCrossingS9N2-v0 --model crossing_task_B_1 --save-interval 100 --frames 5000000  --seed 10 > result/crossing_task_A
python3 -m scripts.train --algo ppo --env MiniGrid-LavaCrossingS9N2-v0 --model crossing_task_B_2 --save-interval 100 --frames 5000000  --seed 11 > result/crossing_task_A
python3 -m scripts.train --algo ppo --env MiniGrid-LavaCrossingS9N2-v0 --model crossing_task_B_3 --save-interval 100 --frames 5000000  --seed 12 > result/crossing_task_A

