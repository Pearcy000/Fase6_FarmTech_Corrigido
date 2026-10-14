"""Paths and CPU protocol shared by all project scripts."""
import os
from pathlib import Path
PROJECT = Path(os.environ.get('FARMTECH_PROJECT', Path(__file__).resolve().parent)).resolve()
ROOT = Path(os.environ.get('FARMTECH_DATA', PROJECT / 'openimages')).resolve()
YOLO = Path(os.environ.get('YOLO_REPO', PROJECT / 'yolov5')).resolve()
DEVICE = 'cpu'
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')
os.environ.setdefault('MPLBACKEND', 'Agg')
# Official PyTorch compatibility option for trusted legacy YOLO module checkpoints.
# Use only the supplied project checkpoints and official Ultralytics weights.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
