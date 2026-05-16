from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List

import cv2

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


def merge_video(fps: float, frames_pattern: str, temp_video: str) -> bool:
	cmd = ['-hwaccel', 'auto', '-r', str(fps), '-i', frames_pattern, '-c:v', config.OUTPUT_VIDEO_ENCODER]
	if config.OUTPUT_VIDEO_ENCODER in ('libx264', 'libx265'):
		crf = round(51 - (config.OUTPUT_VIDEO_QUALITY * 0.51))
		cmd.extend(['-crf', str(crf)])
	if config.OUTPUT_VIDEO_ENCODER == 'libvpx-vp9':
		crf = round(63 - (config.OUTPUT_VIDEO_QUALITY * 0.63))
		cmd.extend(['-crf', str(crf)])
	if config.OUTPUT_VIDEO_ENCODER in ('h264_nvenc', 'hevc_nvenc'):
		crf = round(51 - (config.OUTPUT_VIDEO_QUALITY * 0.51))
		cmd.extend(['-cq', str(crf)])
	cmd.extend(['-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-y', temp_video])
	return _run_ffmpeg(cmd)


def restore_audio(temp_video: str, original_video: str, output_path: str) -> bool:
	cmd = ['-hwaccel', 'auto', '-i', temp_video, '-i', original_video, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y', output_path]
	return _run_ffmpeg(cmd)


def clear_temp_dir(temp_dir: Path) -> None:
	if temp_dir.is_dir():
		shutil.rmtree(temp_dir, ignore_errors=True)
