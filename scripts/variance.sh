#! /bin/bash
#SBATCH -t 3-00:00
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

for i in {0..5}; do
  python3 train/train.py --epochs 1000 --random_seed "$i" || exit
done
