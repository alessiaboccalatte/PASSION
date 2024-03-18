#!/bin/bash

source /scratch/clear/aboccala/anaconda3/etc/profile.d/conda.sh
conda activate py_passion
export CUDA_VISIBLE_DEVICES=0
python /scratch/clear/aboccala/PASSION/workflow/scripts/train_section_segmentation.py