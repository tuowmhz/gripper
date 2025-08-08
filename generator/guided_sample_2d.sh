#!/bin/bash
set -e

# Run the preprocessing pipeline first
/home/rzhao/GripperDesign/SoftFingerDemo2/SoftFingerDemo2/run_pipeline.sh

# Find next available run number in models/2d/
base_dir="models/2d"
prefix="refinedMaskTask"
existing=($(ls "$base_dir" | grep "^$prefix[0-9]\+" | sed -E "s/^$prefix([0-9]+).*/\1/" | sort -n))

if [ ${#existing[@]} -eq 0 ]; then
    next_id=0
else
    last_id=${existing[-1]}
    next_id=$((last_id + 1))
fi

save_path="$base_dir/${prefix}${next_id}"
echo "🔧 Saving to: $save_path"

# Create the directory
mkdir -p "$save_path"

# Run the Python generation script
python generator/train.py \
    --mode='test' \
    --checkpoint_path='ckpts/dynamics_2d.pt' \
    --classifier_guidance \
    --diffusion_checkpoint_path='ckpts/diffusion_2d.ckpt' \
    --object_dir='/home/rzhao/GripperDesign/SoftFingerDemo2/SoftFingerDemo2/refined_mask.npy' \
    --save_dir="$save_path" \
    --ctrlpts_dim=14 \
    --num_fingers=2 \
    --grid_size=360 \
    --num_pos=5 \
    --object_max_num_vertices=100 \
    --num_workers=0 \
    --num_train_timesteps=15 \
    --num_inference_steps=5 \
    --ema_power=0.85 \
    --batch_size=2 \
    --num_cpus=8 \
    --seed=2


# python generator/train.py --mode='test' --checkpoint_path='ckpts/dynamics_2d.pt' \
#     --classifier_guidance --diffusion_checkpoint_path='ckpts/diffusion_2d.ckpt' --object_dir='data/YellowTriangle.npy' --save_dir='models/2d/YELLOWTRIANGLE2'\
#     --ctrlpts_dim=14 --num_fingers=2 --grid_size=360 --num_pos=5 --object_max_num_vertices=100 \
#     --num_workers=0 --num_train_timesteps=15 --num_inference_steps=5 --ema_power=0.85 --batch_size=2  --num_cpus=8 --seed=2