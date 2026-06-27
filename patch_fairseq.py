#!/usr/bin/env python3
"""修复 Python 3.12 兼容性：fairseq + hydra-core + omegaconf"""
import os, re, sys, glob


def patch_file(filepath, pattern, replacement):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    if pattern(content):
        return False
    new = replacement(content)
    if new != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new)
        return True
    return False


def find_pkg(pkg_name):
    for p in sys.path:
        d = os.path.join(p, pkg_name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "__init__.py")):
            return d
    return None


# 1. 修复 fairseq/__init__.py：跳过 hydra_init 和繁重导入
pkg_dir = find_pkg("fairseq")
if pkg_dir:
    fp = os.path.join(pkg_dir, "__init__.py")
    with open(fp, "r") as f:
        content = f.read()
    # 替换 hydra_init 部分
    old = (
        "# initialize hydra\n"
        "from fairseq.dataclass.initialize import hydra_init\n\n"
        "hydra_init()\n\n"
        "import fairseq.criterions  # noqa\n"
        "import fairseq.distributed  # noqa\n"
        "import fairseq.models  # noqa\n"
        "import fairseq.modules  # noqa\n"
        "import fairseq.optim  # noqa\n"
        "import fairseq.optim.lr_scheduler  # noqa\n"
        "import fairseq.pdb  # noqa\n"
        "import fairseq.scoring  # noqa"
    )
    new = (
        "# minimal init - patched for Python 3.12 compatibility\n"
        "import fairseq.models  # noqa"
    )
    if old in content:
        content = content.replace(old, new, 1)
        with open(fp, "w") as f:
            f.write(content)
        print("✓ 修复 fairseq/__init__.py (跳过 hydra_init)")
    else:
        print("⚠️ fairseq/__init__.py 已打补丁或模式不匹配")
else:
    print("⚠️ fairseq 未找到")

# 2. 修复所有 dataclass 中的 mutable default
def fix_dataclass_defaults(pkg_name):
    pkg_dir = find_pkg(pkg_name)
    if not pkg_dir:
        return 0
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
        content = re.sub(r"field\(default=(\w+)\(\)\)", r"field(default_factory=\1)", content)
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
                patched += 1
            except:
                pass
    if patched:
        print(f"✓ {pkg_name}: 修复 {patched} 个文件中的 dataclass")
    return patched


fix_dataclass_defaults("fairseq")
fix_dataclass_defaults("hydra")
fix_dataclass_defaults("omegaconf")

print("✅ 补丁完成")
