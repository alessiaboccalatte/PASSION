#!/bin/bash

source /scratch/clear/aboccala/anaconda3/etc/profile.d/conda.sh
conda activate py_passion
source gpu_setVisibleDevices.sh
python /scratch/clear/aboccala/PASSION/workflow/scripts/train_section_segmentation.py