from __future__ import annotations

import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'faceswap_pack'

MODEL_RETINA = MODEL_DIR / 'retinaface_10g.onnx'
MODEL_ARCFACE = MODEL_DIR / 'arcface_w600k_r50.onnx'
MODEL_INSWAPPER = MODEL_DIR / 'inswapper_128.onnx'
MODEL_FACE_PARSER = MODEL_DIR / 'face_parser.onnx'

TEMP_ROOT = Path(tempfile.gettempdir()) / 'facetime_standalone'

FACE_DETECTOR_SIZE = '320x320'
FACE_DETECTOR_SCORE = 0.5
FACE_ANALYSER_ORDER = 'left-right'

FACE_MASK_BLUR = 0.3
FACE_MASK_PADDING = (0, 0, 0, 0)
FACE_MASK_REGIONS = [
	'skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye',
	'eye-glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip',
]

TEMP_FRAME_FORMAT = 'jpg'
TEMP_FRAME_QUALITY = 100
OUTPUT_VIDEO_ENCODER = 'libx264'
OUTPUT_VIDEO_QUALITY = 80
OUTPUT_VIDEO_NAME = 'temp_merged.mp4'
