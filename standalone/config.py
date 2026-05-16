from __future__ import annotations

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'faceswap_pack'

MODEL_RETINA = MODEL_DIR / 'retinaface_10g.onnx'
MODEL_ARCFACE = MODEL_DIR / 'arcface_w600k_r50.onnx'
MODEL_INSWAPPER = MODEL_DIR / 'inswapper_128.onnx'
MODEL_FACE_PARSER = MODEL_DIR / 'face_parser.onnx'

TEMP_ROOT = Path(tempfile.gettempdir()) / 'facetime_standalone'

# RetinaFace ONNX（faceswap 常见导出）按 640 输入声明中间输出形状；用 320 时易出现 ORT VerifyOutputSizes 告警。
# 需要与 FaceFusion 原版一致时可设环境变量：FACETIME_DETECTOR_SIZE=320x320
FACE_DETECTOR_SIZE = os.environ.get('FACETIME_DETECTOR_SIZE', '640x640')
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


def ort_session_options():
	import onnxruntime as ort

	so = ort.SessionOptions()
	if os.environ.get('FACETIME_ORT_VERBOSE', '').lower() not in ('1', 'true', 'yes'):
		so.log_severity_level = 3
	return so
