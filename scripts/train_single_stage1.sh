srun -p Pixel -x SH-IDC2-172-20-20-[65,66,68,69,71,74,77,86,92,96,97] --mpi=pmi2 --job-name=train1_single --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 --quotatype=spot \
python -u basicsr/train.py -opt options/single_stage1.yml --launcher="slurm"