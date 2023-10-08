srun -p pixel --mpi=pmi2 --job-name=single_naf --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 --quotatype=spot \
python -u basicsr/train.py -opt options/single_naf.yml --launcher="slurm"