#!/usr/bin/env python3
"""安装后修复 — 修复 fairseq + hydra-core 的 Python 3.12 兼容性"""
import os, re, sys, glob

def patch_package(pkg_name):
    """查找并修复指定包中的 dataclass mutable default 问题"""
    pkg_dir = None
    for p in sys.path:
        d = os.path.join(p, pkg_name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "__init__.py")):
            pkg_dir = d
            break

    if not pkg_dir:
        print(f"⚠️ {pkg_name} 未找到，跳过")
        return 0

    print(f"✓ {pkg_name}: {pkg_dir}")
    patched = 0
    for filepath in glob.glob(os.path.join(pkg_dir, "**/*.py"), recursive=True):
        if "__pycache__" in filepath:
            continue
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except:
            continue
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
                content = re.sub(
                    r"from dataclasses import(.*)",
                    r"from dataclasses import field,\1",
                    content,
                    count=1,
                )
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                rel = os.path.relpath(filepath, pkg_dir)
                if patched < 10:
                    print(f"  ✓ 修复: {rel}")
                patched += 1
            except Exception as e:
                print(f"  ✗ 写入失败 {filepath}: {e}")

    return patched


total = 0
print("=" * 50)
print("  修复 Python 3.12 兼容性")
print("=" * 50)
for pkg in ["fairseq", "hydra", "hydra_core"]:
    total += patch_package(pkg)
print("=" * 50)
print(f"✅ 总计修复 {total} 个文件")
