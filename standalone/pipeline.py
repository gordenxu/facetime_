from __future__ import annotations

import glob
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

from . import config
from . import detect
from . import ffmpeg_util
from . import inswap
from . import mask


def _normalize_output_path(
	source_paths: List[str], target_path: str, output_path: str
) -> str | None:
	def is_file(p: str) -> bool:
		return bool(p and os.path.isfile(p))

	def is_dir(p: str) -> bool:
		return bool(p and os.path.isdir(p))

	tname = os.path.splitext(os.path.basename(target_path))[0]
	text = os.path.splitext(os.path.basename(target_path))[1]
	if is_file(target_path) and is_dir(output_path):
		if source_paths and is_file(source_paths[0]):
			sname = os.path.splitext(os.path.basename(source_paths[0]))[0]
			return os.path.join(output_path, sname + '-' + tname + text)
		return os.path.join(output_path, tname + text)
	if is_file(target_path) and output_path:
		odir = os.path.dirname(output_path)
		oname, oext = os.path.splitext(os.path.basename(output_path))
		if is_dir(odir) and oext:
			return os.path.join(odir, oname + text)
		return None
	return output_path


def _pick_providers() -> List[str]:
	import onnxruntime

	avail = onnxruntime.get_available_providers()
	# 默认不包含 TensorRT：多数环境未安装 libnvinfer，会刷屏报错后仍回退 CUDA。
	# 已安装 TensorRT 并配置好 LD_LIBRARY_PATH 时设：FACETIME_ORT_TENSORRT=1
	pref: List[str] = []
	if os.environ.get('FACETIME_ORT_TENSORRT', '').lower() in ('1', 'true', 'yes'):
		pref.append('TensorrtExecutionProvider')
	pref.extend(
		[
			'CUDAExecutionProvider',
			'CoreMLExecutionProvider',
			'CPUExecutionProvider',
		]
	)
	chosen = [p for p in pref if p in avail]
	return chosen or list(avail)


def _temp_dir(target_path: str) -> Path:
	stem = Path(target_path).stem
	d = config.TEMP_ROOT / stem
	d.mkdir(parents=True, exist_ok=True)
	return d


def _frame_paths(temp_dir: Path) -> List[str]:
	pat = str(temp_dir / ('*.' + config.TEMP_FRAME_FORMAT))
	return sorted(glob.glob(pat))


def run(
	source_paths: List[str],
	target_path: str,
	output_path: str,
	*,
	keep_fps: bool = False,
	skip_audio: bool = False,
	thread_count: int = 10,
) -> bool:
	out = _normalize_output_path(source_paths, target_path, output_path)
	if not out:
		print('无效的输出路径')
		return False
	os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

	for m in (
		config.MODEL_RETINA,
		config.MODEL_ARCFACE,
		config.MODEL_INSWAPPER,
		config.MODEL_FACE_PARSER,
	):
		if not m.is_file():
			print(f'缺少模型文件: {m}')
			return False

	providers = _pick_providers()
	detect.init_sessions(providers)
	mask.init_face_parser(providers)
	inswap.init_inswapper(providers)

	try:
		source_face = inswap.load_source_face(source_paths)
	except Exception as exc:  # noqa: BLE001
		print(exc)
		return False

	fps = ffmpeg_util.detect_fps(target_path) if keep_fps else 25.0
	if not fps:
		fps = 25.0

	temp_dir = _temp_dir(target_path)
	frames_pattern = str(temp_dir / ('%04d.' + config.TEMP_FRAME_FORMAT))
	temp_video = str(temp_dir / config.OUTPUT_VIDEO_NAME)

	try:
		if not ffmpeg_util.extract_frames(target_path, fps, frames_pattern):
			print('抽帧失败')
			return False

		paths = _frame_paths(temp_dir)
		if not paths:
			print('未找到临时帧')
			return False

		def work_one(p: str) -> None:
			frame = detect.read_static_image(p)
			if frame is None:
				return
			new_frame = inswap.process_frame(source_face, frame)
			cv2.imwrite(p, new_frame)

		with ThreadPoolExecutor(max_workers=thread_count) as ex:
			futures = [ex.submit(work_one, p) for p in paths]
			for _ in tqdm(as_completed(futures), total=len(futures), desc='换脸', unit='帧'):
				_.result()

		if not ffmpeg_util.merge_video(fps, frames_pattern, temp_video, total_frames=len(paths)):
			print('合成视频失败')
			return False

		if skip_audio:
			if Path(out).exists():
				Path(out).unlink()
			shutil.move(temp_video, out)
		else:
			if not ffmpeg_util.restore_audio(temp_video, target_path, out):
				if Path(out).exists():
					Path(out).unlink()
				shutil.move(temp_video, out)

	finally:
		ffmpeg_util.clear_temp_dir(temp_dir)

	print('完成:', out)
	return True


def main(
	source_paths: List[str],
	target_path: str,
	output_path: str,
	**kw: bool | int,
) -> None:
	ok = run(source_paths, target_path, output_path, **kw)
	if not ok:
		raise SystemExit(1)
