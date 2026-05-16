from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import cv2
import numpy
import onnxruntime

from . import config
from .geometry import (
	apply_nms,
	create_static_anchors,
	distance_to_bbox,
	distance_to_kps,
	warp_face,
)
from .types import Embedding, Face, Frame, Kps

_ORT_SEM = threading.Semaphore(1)
_SESSION_RETINA: onnxruntime.InferenceSession | None = None
_SESSION_ARC: onnxruntime.InferenceSession | None = None


def read_static_image(path: str) -> Optional[Frame]:
	if not path:
		return None
	return cv2.imread(path)


def init_sessions(providers: List[str]) -> None:
	global _SESSION_RETINA, _SESSION_ARC
	if _SESSION_RETINA is None:
		_SESSION_RETINA = onnxruntime.InferenceSession(
			str(config.MODEL_RETINA), providers=providers
		)
	if _SESSION_ARC is None:
		_SESSION_ARC = onnxruntime.InferenceSession(
			str(config.MODEL_ARCFACE), providers=providers
		)


def _resize_frame(frame: Frame, max_w: int, max_h: int) -> Frame:
	h, w = frame.shape[:2]
	if h > max_h or w > max_w:
		scale = min(max_h / h, max_w / w)
		return cv2.resize(frame, (int(w * scale), int(h * scale)))
	return frame


def _calc_embedding(temp_frame: Frame, kps: Kps) -> Tuple[Embedding, Embedding]:
	face_recognizer = _SESSION_ARC
	assert face_recognizer is not None
	crop_frame, _ = warp_face(temp_frame, kps, 'arcface_112_v2', (112, 112))
	crop_frame = crop_frame.astype(numpy.float32) / 127.5 - 1
	crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
	crop_frame = numpy.expand_dims(crop_frame, axis=0)
	with _ORT_SEM:
		embedding = face_recognizer.run(
			None, {face_recognizer.get_inputs()[0].name: crop_frame}
		)[0]
	embedding = embedding.ravel()
	normed = embedding / numpy.linalg.norm(embedding)
	return embedding, normed


def _detect_retina(
	temp_frame: Frame,
	temp_h: int,
	temp_w: int,
	det_h: int,
	det_w: int,
	ratio_h: float,
	ratio_w: float,
) -> Tuple[List[numpy.ndarray], List[numpy.ndarray], List[float]]:
	face_detector = _SESSION_RETINA
	assert face_detector is not None
	bbox_list: List[numpy.ndarray] = []
	kps_list: List[numpy.ndarray] = []
	score_list: List[float] = []
	feature_strides = [8, 16, 32]
	feature_map_channel = 3
	anchor_total = 2
	prepare = numpy.zeros((det_h, det_w, 3), dtype=numpy.uint8)
	prepare[:temp_h, :temp_w, :] = temp_frame
	inp = (prepare.astype(numpy.float32) - 127.5) / 128.0
	inp = numpy.expand_dims(inp.transpose(2, 0, 1), axis=0).astype(numpy.float32)
	with _ORT_SEM:
		detections = face_detector.run(None, {face_detector.get_inputs()[0].name: inp})
	for index, feature_stride in enumerate(feature_strides):
		keep = numpy.where(detections[index] >= config.FACE_DETECTOR_SCORE)[0]
		if keep.any():
			stride_h = det_h // feature_stride
			stride_w = det_w // feature_stride
			anchors = create_static_anchors(feature_stride, anchor_total, stride_h, stride_w)
			bbox_raw = detections[index + feature_map_channel] * feature_stride
			kps_raw = detections[index + feature_map_channel * 2] * feature_stride
			for bbox in distance_to_bbox(anchors, bbox_raw)[keep]:
				bbox_list.append(
					numpy.array([
						bbox[0] * ratio_w,
						bbox[1] * ratio_h,
						bbox[2] * ratio_w,
						bbox[3] * ratio_h,
					])
				)
			for kps in distance_to_kps(anchors, kps_raw)[keep]:
				kps_list.append(kps * [ratio_w, ratio_h])
			for score in detections[index][keep]:
				score_list.append(float(score[0]))
	return bbox_list, kps_list, score_list


def extract_faces(frame: Frame) -> List[Face]:
	dw, dh = map(int, config.FACE_DETECTOR_SIZE.split('x'))
	fh, fw, _ = frame.shape
	temp = _resize_frame(frame, dw, dh)
	th, tw, _ = temp.shape
	rh, rw = fh / th, fw / tw
	bbox_list, kps_list, score_list = _detect_retina(temp, th, tw, dh, dw, rh, rw)
	faces: List[Face] = []
	if config.FACE_DETECTOR_SCORE > 0 and bbox_list:
		sort_ix = numpy.argsort(-numpy.array(score_list))
		bbox_list = [bbox_list[i] for i in sort_ix]
		kps_list = [kps_list[i] for i in sort_ix]
		score_list = [score_list[i] for i in sort_ix]
		for index in apply_nms(bbox_list, 0.4):
			kps = kps_list[index]
			emb, normed = _calc_embedding(frame, kps)
			faces.append(
				Face(
					bbox=bbox_list[index],
					kps=kps,
					score=score_list[index],
					embedding=emb,
					normed_embedding=normed,
				)
			)
	return faces


def sort_faces(faces: List[Face]) -> List[Face]:
	order = config.FACE_ANALYSER_ORDER
	if order == 'left-right':
		return sorted(faces, key=lambda f: f.bbox[0])
	if order == 'right-left':
		return sorted(faces, key=lambda f: f.bbox[0], reverse=True)
	if order == 'top-bottom':
		return sorted(faces, key=lambda f: f.bbox[1])
	if order == 'bottom-top':
		return sorted(faces, key=lambda f: f.bbox[1], reverse=True)
	if order == 'small-large':
		return sorted(
			faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
		)
	if order == 'large-small':
		return sorted(
			faces,
			key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
			reverse=True,
		)
	if order == 'best-worst':
		return sorted(faces, key=lambda f: f.score, reverse=True)
	if order == 'worst-best':
		return sorted(faces, key=lambda f: f.score)
	return faces


def get_many_faces(frame: Frame) -> List[Face]:
	return sort_faces(extract_faces(frame))


def get_one_face(frame: Frame, position: int = 0) -> Optional[Face]:
	many = get_many_faces(frame)
	if many:
		try:
			return many[position]
		except IndexError:
			return many[-1]
	return None


def get_average_face(frames: List[Frame], position: int = 0) -> Optional[Face]:
	faces: List[Face] = []
	emb_list: List[numpy.ndarray] = []
	normed_list: List[numpy.ndarray] = []
	for fr in frames:
		f = get_one_face(fr, position)
		if f:
			faces.append(f)
			emb_list.append(f.embedding)
			normed_list.append(f.normed_embedding)
	if not faces:
		return None
	return Face(
		bbox=faces[0].bbox,
		kps=faces[0].kps,
		score=faces[0].score,
		embedding=numpy.mean(emb_list, axis=0),
		normed_embedding=numpy.mean(normed_list, axis=0),
	)
