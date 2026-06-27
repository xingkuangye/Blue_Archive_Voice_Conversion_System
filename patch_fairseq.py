#!/usr/bin/env python3
"""安装后修复 — 在导入 fairseq 之前先打补丁"""
import os, re, sys, glob

# 查找 fairseq 安装路径（不导入）
pkg_dir = None
for p in sys.path:
    d = os.path.join(p, "fairseq")
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, "__init__.py")):
        pkg_dir = d
        break

# 没有通过 pip 安装的 fairseq，检查本地源码安装
if not pkg_dir:
    for p in sys.path:
        for d_name in os.listdir(p):
            if "fairseq" in d_name and os.path.isdir(os.path.join(p, d_name)):
                d = os.path.join(p, d_name)
                if os.path.isfile(os.path.join(d, "fairseq", "__init__.py")):
                    pkg_dir = os.path.join(d, "fairseq")
                elif os.path.isfile(os.path.join(d, "__init__.py")):
                    pkg_dir = d

if not pkg_dir:
    print("❌ fairseq 未找到，请确保已安装")
    print("查找路径:", sys.path)
    sys.exit(1)

print(f"✓ fairseq 路径: {pkg_dir}")

# 修复所有 .py 文件中的 dataclass mutable default 问题
patched = 0
for filepath in glob.glob(os.path.join(pkg_dir, "**/*.py"), recursive=True):
    if "__pycache__" in filepath:
        continue
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    original = content

    # 修复 `field(default=Xxx())` → `field(default_factory=Xxx)`
    content = re.sub(r"field\(default=(\w+)\(\)\)", r"field(default_factory=\1)", content)

    # 修复 `xxx: XxxConfig = XxxConfig()` → `xxx: XxxConfig = field(default_factory=XxxConfig)`
    content = re.sub(
        r"^(\s+)(\w+): (\w+) = \3\(\)",
        r"\1\2: \3 = field(default_factory=\3)",
        content,
        flags=re.MULTILINE,
    )

    if content != original:
        if "from dataclasses import field" not in content:
            # 确保 field 已导入
            content = re.sub(
                r"from dataclasses import(.*)",
                r"from dataclasses import field,\1",
                content,
                count=1,
            )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        rel = os.path.relpath(filepath, pkg_dir)
        if patched < 3 or "configs" in rel:
            print(f"  ✓ 修复: {rel}")
        patched += 1

print(f"\n✅ 已修复 {patched} 个文件")
