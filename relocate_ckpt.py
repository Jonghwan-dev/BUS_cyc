#!/usr/bin/env python3
"""
Move CycleGAN best checkpoints into compact folders.

Usage:
    python relocate_ckpt.py --ckpt_root ./checkpoints/sweep_run_V
"""

import argparse, re, shutil, hashlib
from pathlib import Path

# ────────── 정규식 ──────────
#  best_<하이퍼파라미터...>_net_G_A.pth  (또는 B)
PAT = re.compile(r'(best_.*)_net_G_[AB]\.pth$')

def relocate(root: Path):
    moved = 0
    # rglob → 하위 폴더까지 전부 검색
    for p in root.rglob('best_*_net_G_A.pth'):
        m = PAT.match(p.name)
        if not m:
            print(f"⚠️  regex miss: {p.name}")
            continue
        prefix = m.group(1)                                  # best_<…>
        tag    = hashlib.md5(prefix.encode()).hexdigest()[:8]
        dst    = root / f"best_{tag}"
        dst.mkdir(parents=True, exist_ok=True)

        for suffix in ['net_G_A.pth', 'net_G_B.pth']:
            src = p.with_name(f"{prefix}_{suffix}")
            if not src.exists():
                print(f"❌ missing pair file: {src}")
                break
            shutil.move(src, dst / suffix)
        else:
            # 두 파일을 모두 옮겼을 때만 meta 작성
            (dst / 'meta.txt').write_text(prefix + '\n')
            print(f"✓ {prefix}  →  {dst.name}/")
            moved += 1
    if moved == 0:
        print("🚫 Nothing relocated. 경로·파일명을 다시 확인하세요.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_root', required=True,
                    help='폴더 경로 (예: ./checkpoints/sweep_run_V)')
    args = ap.parse_args()
    root = Path(args.ckpt_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"❌ Directory not found: {root}")
    relocate(root)

if __name__ == '__main__':
    main()