"""
硬件配置 — 强制 CPU 推理
"""
import argparse
import sys
import torch
from multiprocessing import cpu_count


class Config:
    def __init__(self):
        # 强制 CPU 推理
        self.device = "cpu"
        self.is_half = False  # CPU 不支持 half precision
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        (
            self.colab,
            self.api,
            self.unsupported
        ) = self.arg_parse()
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def arg_parse() -> tuple:
        parser = argparse.ArgumentParser()
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument("--api", action="store_true", help="Launch with api")
        parser.add_argument("--unsupported", action="store_true", help="Enable unsupported feature")
        cmd_opts = parser.parse_args()
        return (
            cmd_opts.colab,
            cmd_opts.api,
            cmd_opts.unsupported
        )

    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def device_config(self) -> tuple:
        # 始终使用 CPU
        self.device = "cpu"
        self.is_half = False
        self.gpu_name = None
        self.gpu_mem = None

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        # CPU 配置
        x_pad = 1
        x_query = 6
        x_center = 38
        x_max = 41

        return x_pad, x_query, x_center, x_max
