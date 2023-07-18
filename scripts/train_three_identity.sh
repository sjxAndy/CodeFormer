srun -p Pixel -x SH-IDC2-172-20-20-[65,66,68,71,86,96] --mpi=pmi2 --job-name=train_id --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 \
python -u basicsr/train.py -opt options/CodeFormer_stage3_identity.yml --launcher="slurm"
