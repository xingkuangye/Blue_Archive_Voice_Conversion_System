#!/usr/bin/env python3
"""
Blue Archive RVC — 入口
"""
import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.api import run

if __name__ == "__main__":
    run()
