from __future__ import annotations

from typing import Any, List

import numpy
import onnx
import onnxruntime
from onnx import numpy_helper

from . import config
from .detect import _ORT_SEM, get_average_face, get_one_face, read_static_image
from .geometry import paste_back, warp_face
from .mask import create_region_mask, create_static_box_mask
from .types import Face, Frame

_SESSION_SWAP: onnxruntime.InferenceSession | None = None
_MODEL_MATRIX: numpy.ndarray[Any, Any] | None = None

INSWAPPER_SIZE: tuple[int, int] = (128, 128)
INSWAPPER_TEMPLATE = 'arcface_128_v2'
MODEL_MEAN = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float32)
MODEL_STD = numpy.array([1.0, 1.0, 1.0], dtype=numpy.float32)


def init_inswapper(providers: List[str]) -> None:
	global _SESSION_SWAP, _MODEL_MATRIX
	if _SESSION_SWAP is None:
		_SESSION_SWAP = onnxruntime.InferenceSession(
			str(config.MODEL_INSWAPPER),
			providers=providers,
			sess_options=config.ort_session_options(),
		)
	if _MODEL_MATRIX is None:
		model = onnx.load(str(config.MODEL_INSWAPPER))
		_MODEL_MATRIX = numpy_helper.to_array(model.graph.initializer[-1])


def _prepare_source_embedding(source_face: Face) -> numpy.ndarray:
	assert _MODEL_MATRIX is not None
	se = source_face.embedding.reshape((1, -1))
	se = numpy.dot(se, _MODEL_MATRIX) / numpy.linalg.norm(se)
	return se.astype(numpy.float32)


def _prepare_crop(crop: Frame) -> numpy.ndarray:
	x = crop[:, :, ::-1].astype(numpy.float32) / 255.0
	x = (x - MODEL_MEAN) / MODEL_STD
	x = x.transpose(2, 0, 1)
	return numpy.expand_dims(x, axis=0).astype(numpy.float32)


def _normalize_crop(crop: numpy.ndarray) -> Frame:
	x = crop.transpose(1, 2, 0)
	x = (x * 255.0).round()
	return x[:, :, ::-1].astype(numpy.uint8)


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
	frame_processor = _SESSION_SWAP
	assert frame_processor is not None
	crop_frame, affine = warp_face(
		temp_frame, target_face.kps, INSWAPPER_TEMPLATE, INSWAPPER_SIZE
	)
	crop_masks: List[numpy.ndarray] = []
	crop_masks.append(
		create_static_box_mask(
			crop_frame.shape[:2][::-1],
			config.FACE_MASK_BLUR,
			config.FACE_MASK_PADDING,
		)
	)
	prep = _prepare_crop(crop_frame)
	inp_src = _prepare_source_embedding(source_face)
	inputs: dict[str, Any] = {}
	for inp in frame_processor.get_inputs():
		if inp.name == 'source':
			inputs[inp.name] = inp_src
		elif inp.name == 'target':
			inputs[inp.name] = prep
	with _ORT_SEM:
		out = frame_processor.run(None, inputs)[0]
	crop_out = _normalize_crop(out[0])
	crop_masks.append(
		create_region_mask(crop_out, list(config.FACE_MASK_REGIONS))
	)
	crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
	return paste_back(temp_frame, crop_out, crop_mask, affine)


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
	target = get_one_face(temp_frame, 0)
	if not target:
		return temp_frame
	return swap_face(source_face, target, temp_frame)


def load_source_face(source_paths: List[str]) -> Face:
	frames = []
	for p in source_paths:
		f = read_static_image(p)
		if f is None:
			raise FileNotFoundError(p)
		frames.append(f)
	sf = get_average_face(frames, 0)
	if sf is None:
		raise RuntimeError('源图中未检测到人脸')
	return sf
