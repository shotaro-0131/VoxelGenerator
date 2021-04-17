#!/bin/sh
#$ -cwd
# 資源タイプF 1ノードを使用
#$ -l q_node=1
#$ -l h_rt=0:10:00
#$ -N grid
#$ -o out.txt
#$ -e error.txt
#$ -m a
#$ -m b
#$ -m e
#$ -M murata.s.ah@m.titech.ac.jp
. /etc/profile.d/modules.sh
module load python/3.6.5
module load cuda cudnn openmpi
module load intel
python3 -m pip install --user -r requirements.txt

python train.py