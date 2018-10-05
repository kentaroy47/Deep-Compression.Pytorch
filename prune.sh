export CUDA_VISUAL_DEVICES=0

python prune.py --prune 0.1
python prune.py --prune 0.25
python prune.py --prune 0.5
python prune.py --prune 0.75