
from __future__ import annotations

import argparse
import sys


# 未在命令行指定时使用的默认路径（与旧版常量行为一致）
_DEFAULT_SOURCES = ['source.png']
_DEFAULT_TARGET = 'target.mp4'
_DEFAULT_OUTPUT = 'output.mp4'


def _build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description='独立换脸管线：视频目标 + 源人脸图 → 输出视频。',
	)
	p.add_argument(
		'-s',
		'--source',
		action='append',
		dest='source_paths',
		metavar='PATH',
		help='源人脸图片，可多次传入以多人平均embedding（默认: %s）' % _DEFAULT_SOURCES[0],
	)
	p.add_argument(
		'-t',
		'--target',
		default=_DEFAULT_TARGET,
		help='目标视频路径（默认: %s）' % _DEFAULT_TARGET,
	)
	p.add_argument(
		'-o',
		'--output',
		default=_DEFAULT_OUTPUT,
		help='输出视频路径，或已存在目录（默认: %s）' % _DEFAULT_OUTPUT,
	)
	p.add_argument(
		'--keep-fps',
		action='store_true',
		help='保留目标视频帧率抽帧（默认按 25fps）',
	)
	p.add_argument(
		'--skip-audio',
		action='store_true',
		help='输出不包含音轨（不尝试从原视频拷贝音频）',
	)
	p.add_argument(
		'-j',
		'--threads',
		type=int,
		default=10,
		metavar='N',
		help='并行处理帧的线程数（默认: 10）',
	)
	return p


def main(argv: list[str] | None = None) -> None:
	args = _build_parser().parse_args(argv)
	from .pipeline import run

	sources = args.source_paths if args.source_paths else list(_DEFAULT_SOURCES)
	ok = run(
		sources,
		args.target,
		args.output,
		keep_fps=args.keep_fps,
		skip_audio=args.skip_audio,
		thread_count=args.threads,
	)
	sys.exit(0 if ok else 1)


if __name__ == '__main__':
	main()
