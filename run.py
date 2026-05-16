#!/usr/bin/env python3
"""仓库根目录运行；参数会传给 standalone，例如：
python run.py -s a.png -t in.mp4 -o out.mp4 --keep-fps
"""
import runpy
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
	sys.path.insert(0, str(_REPO))

if __name__ == '__main__':
	runpy.run_module('standalone', run_name='__main__')