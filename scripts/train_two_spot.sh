srun -p Pixel -x SH-IDC2-172-20-20-[65,86] --mpi=pmi2 --job-name=train2 --gres=gpu:1 --ntasks=8 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 --quotatype=spot \
python -u basicsr/train.py -opt options/CodeFormer_stage2.yml --launcher="slurm"
