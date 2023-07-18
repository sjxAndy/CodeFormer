srun -p Pixel -x SH-IDC2-172-20-20-[65,86] --mpi=pmi2 --job-name=train2 --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 \
python -u basicsr/train.py -opt options/CodeFormer_stage2_resume.yml --launcher="slurm"
