#!/usr/bin/env python3
"""Wrapper to start YOLO26m training pipeline."""
import sys
sys.path.insert(0, '/home/jovyan/shared/s0598584/deepseek/scripts')
from train_pipeline import run_full, make_spec
spec = make_spec(
    name='yolo26m_face_lp_640',
    data='/home/jovyan/shared/s0598584/deepseek/dataset_face_lp_640/dataset.yaml',
    imgsz=640,
    model_size='m',
)
run_full(spec)