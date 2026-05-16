from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

import cv2
import numpy
import onnxruntime

from . import config
from .detect import _ORT_SEM
from .types import FaceMaskRegion, Frame, Mask, Size

_SESSION_PARSER: onnxruntime.InferenceSession | None = None

FACE_MASK_REGIONS: Dict[FaceMaskRegion, int] = {
	'skin': 1,
	'left-eyebrow': 2,
	'right-eyebrow': 3,
	'left-eye': 4,
	'right-eye': 5,
	'eye-glasses': 6,
	'nose': 10,
	'mouth': 11,
	'upper-lip': 12,
	'lower-lip': 13,
}


def init_face_parser(providers: List[str]) -> None:
	global _SESSION_PARSER
	if _SESSION_PARSER is None:
		_SESSION_PARSER = onnxruntime.InferenceSession(
			str(config.MODEL_FACE_PARSER), providers=providers
		)


@lru_cache(maxsize=None)
def create_static_box_mask(crop_size: Size, face_mask_blur: float, padding: tuple) -> Mask:
	blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
	blur_area = max(blur_amount // 2, 1)
	box_mask = numpy.ones(crop_size, numpy.float32)
	box_mask[: max(blur_area, int(crop_size[1] * padding[0] / 100)), :] = 0
	box_mask[-max(blur_area, int(crop_size[1] * padding[2] / 100)) :, :] = 0
	box_mask[:, : max(blur_area, int(crop_size[0] * padding[3] / 100))] = 0
	box_mask[:, -max(blur_area, int(crop_size[0] * padding[1] / 100)) :] = 0
	if blur_amount > 0:
		box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
	return box_mask


def create_region_mask(crop_frame: Frame, face_mask_regions: List[FaceMaskRegion]) -> Mask:
	face_parser = _SESSION_PARSER
	assert face_parser is not None
	prepare = cv2.flip(cv2.resize(crop_frame, (512, 512)), 1)
	prepare = numpy.expand_dims(prepare, axis=0).astype(numpy.float32)[:, :, ::-1] / 127.5 - 1
	prepare = prepare.transpose(0, 3, 1, 2)
	with _ORT_SEM:
		region_mask = face_parser.run(
			None, {face_parser.get_inputs()[0].name: prepare}
		)[0][0]
	region_mask = numpy.isin(
		region_mask.argmax(0), [FACE_MASK_REGIONS[r] for r in face_mask_regions]
	)
	return cv2.resize(region_mask.astype(numpy.float32), crop_frame.shape[:2][::-1])
