#!/bin/bash
conda init bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate track_tool
python deploy/run_test.py