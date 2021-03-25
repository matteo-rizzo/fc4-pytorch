#! /bin/bash
#SBATCH -t 2-00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100l:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --account=def-

cd ..

# module load python/3.6

source venv/bin/activate

# export PYTHONPATH=$PYTHONPATH:~/projects/def-conati/marizzo/xai/pytorch-fc4
export PYTHONPATH=$PYTHONPATH:~/home/matteo/Projects/pytorch-fc4

python3 train/train.py --epochs 1000 --random_seed 1 || exit
python3 train/train.py --epochs 1000 --random_seed 2 || exit
python3 train/train.py --epochs 1000 --random_seed 3 || exit
python3 train/train.py --epochs 1000 --random_seed 4 || exit
