from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy

Frame = numpy.ndarray
Mask = numpy.ndarray
Matrix = numpy.ndarray
Bbox = numpy.ndarray
Kps = numpy.ndarray
Score = float
Embedding = numpy.ndarray
Size = tuple[int, int]
Template = Literal['arcface_112_v1', 'arcface_112_v2', 'arcface_128_v2', 'ffhq_512']
FaceMaskRegion = Literal[
	'skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye',
	'eye-glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip',
]


@dataclass
class Face:
	bbox: numpy.ndarray
	kps: numpy.ndarray
	score: float
	embedding: numpy.ndarray
	normed_embedding: numpy.ndarray
