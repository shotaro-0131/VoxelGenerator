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
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export PATH="$PYENV_ROOT/versions/anaconda3-4.0.0/bin:$PATH"


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$($PYENV_ROOT'/versions/anaconda3-4.0.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$PYENV_ROOT'/versions/anaconda3-4.0.0/etc/profile.d/conda.sh" ]; then
        . "$PYENV_ROOT'/.pyenv/versions/anaconda3-4.0.0/etc/profile.d/conda.sh"
    else
        export PATH="$HOME'/.pyenv/versions/anaconda3-4.0.0/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate py36
python3 -m pip install --user -r requirements.txt

python train.py