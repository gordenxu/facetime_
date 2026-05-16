from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

from . import config


def _run_ffmpeg(args: List[str]) -> bool:
	cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', *args]
	try:
		subprocess.run(cmd, stderr=subprocess.PIPE, check=True)
		return True
	except subprocess.CalledProcessError:
		return False


def detect_fps(video_path: str) -> float | None:
	cap = cv2.VideoCapture(video_path)
	if cap.isOpened():
		fps = cap.get(cv2.CAP_PROP_FPS)
		cap.release()
		return float(fps) if fps else None
	return None


def extract_frames(target_path: str, fps: float, frames_pattern: str) -> bool:
	q = round(31 - (config.TEMP_FRAME_QUALITY * 0.31))
	cmd = [
		'-hwaccel',
		'auto',
		'-i',
		target_path,
		'-q:v',
		str(q),
		'-pix_fmt',
		'rgb24',
		'-vf',
		f'fps={fps}',
		'-vsync',
		'0',
		frames_pattern,
	]
	return _run_ffmpeg(cmd)


_FRAME_RE = re.compile(r'frame=\s*(\d+)')


def merge_video(
	fps: float,
	frames_pattern: str,
	temp_video: str,
	*,
	total_frames: int | None = None,
) -> bool:
	enc: List[str] = [
		'-hwaccel',
		'auto',
		'-r',
		str(fps),
		'-i',
		frames_pattern,
		'-c:v',
		config.OUTPUT_VIDEO_ENCODER,
	]
	if config.OUTPUT_VIDEO_ENCODER in ('libx264', 'libx265'):
		crf = round(51 - (config.OUTPUT_VIDEO_QUALITY * 0.51))
		enc.extend(['-crf', str(crf)])
	if config.OUTPUT_VIDEO_ENCODER == 'libvpx-vp9':
		crf = round(63 - (config.OUTPUT_VIDEO_QUALITY * 0.63))
		enc.extend(['-crf', str(crf)])
	if config.OUTPUT_VIDEO_ENCODER in ('h264_nvenc', 'hevc_nvenc'):
		crf = round(51 - (config.OUTPUT_VIDEO_QUALITY * 0.51))
		enc.extend(['-cq', str(crf)])
	enc.extend(['-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-y', temp_video])

	if total_frames is not None and total_frames > 0:
		return _merge_video_with_progress(enc, total_frames)

	cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', *enc]
	return _run_ffmpeg(cmd)


def _merge_video_with_progress(enc_args: List[str], total_frames: int) -> bool:
	cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'info', *enc_args]
	proc = subprocess.Popen(
		cmd,
		stderr=subprocess.PIPE,
		stdout=subprocess.DEVNULL,
		text=False,
	)
	assert proc.stderr is not None
	last_shown = 0
	buf = b''
	try:
		with tqdm(total=total_frames, desc='合成视频', unit='帧', ascii=' =') as bar:
			while True:
				chunk = proc.stderr.read(8192)
				if not chunk:
					break
				buf += chunk
				if len(buf) > 64000:
					buf = buf[-32000:]
				dec = buf.decode('utf-8', errors='replace')
				for m in _FRAME_RE.finditer(dec):
					n = int(m.group(1))
					cap = min(n, total_frames)
					if cap > last_shown:
						bar.update(cap - last_shown)
						last_shown = cap
			proc.wait()
			if proc.returncode == 0 and last_shown < total_frames:
				bar.update(total_frames - last_shown)
	except BaseException:
		proc.kill()
		try:
			proc.wait(timeout=10)
		except Exception:  # noqa: BLE001
			pass
		raise
	return proc.returncode == 0


def restore_audio(temp_video: str, original_video: str, output_path: str) -> bool:
	cmd = ['-hwaccel', 'auto', '-i', temp_video, '-i', original_video, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y', output_path]
	return _run_ffmpeg(cmd)


def clear_temp_dir(temp_dir: Path) -> None:
	if temp_dir.is_dir():
		shutil.rmtree(temp_dir, ignore_errors=True)
