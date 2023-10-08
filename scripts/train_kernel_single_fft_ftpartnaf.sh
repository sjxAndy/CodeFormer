srun -p pixel --mpi=pmi2 --job-name=fft_ftpart --gres=gpu:2 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=5 --kill-on-bad-exit=1 --quotatype=spot \
python -u basicsr/train.py -opt options/Kernel_single_fft_ftpartnaf.yml --launcher="slurm"