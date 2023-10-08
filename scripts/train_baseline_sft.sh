srun -p pixel -w SH-IDC1-10-198-6-168 --mpi=pmi2 --job-name=train_sft --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 --quotatype=spot \
python -u basicsr/train.py -opt options/Baseline-sft-width64.yml --launcher="slurm"