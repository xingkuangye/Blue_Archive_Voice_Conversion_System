#!/usr/bin/env python3
"""安装后修复 — 打补丁修复 fairseq Python 3.12 兼容性"""
import os, re, sys, glob

# 查找 fairseq 安装路径
try:
    import fairseq
    pkg_dir = os.path.dirname(fairseq.__file__)
except ImportError:
    print("fairseq 未安装，跳过补丁")
    sys.exit(0)

print(f"fairseq 路径: {pkg_dir}")

# 修复 dataclass 中的 mutable default 问题
patched = 0
for filepath in glob.glob(os.path.join(pkg_dir, "**/*.py"), recursive=True):
    if "__pycache__" in filepath:
        continue
    with open(filepath, "r") as f:
        content = f.read()
    # 修复 `field(default=Type())` 模式
    new = re.sub(r"field\(default=(\w+)\(\)\)", r"field(default_factory=\1)", content)
    # 修复 `xxx: XxxConfig = XxxConfig()` 模式（不含 field 包装的）
    new = re.sub(
        r"^(\s+)(\w+): (\w+) = \3\(\)",
        r"\1\2: \3 = field(default_factory=\3)",
        new,
        flags=re.MULTILINE,
    )
    if new != content:
        # 确保 field 已导入
        if "from dataclasses import field" not in new:
            new = new.replace("from dataclasses import", "from dataclasses import field,", 1)
        with open(filepath, "w") as f:
            f.write(new)
        print(f"  ✓ 修复: {os.path.relpath(filepath, pkg_dir)}")
        patched += 1

if patched > 0:
    print(f"已修复 {patched} 个文件")
else:
    print("无需修复")
