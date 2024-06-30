#!/bin/bash

python 0-synchronize_Video.py

python 1-cut_Video_into_Imgs.py

python 2-compute_Calibration_Matrix.py

python 3-apply_Calibration_Matrix.py

python 4-compute_Homography_Matrix.py

python 5-transform_Perspective.py