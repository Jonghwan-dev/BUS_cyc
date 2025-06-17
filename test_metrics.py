#!/usr/bin/env python3
"""
CycleGAN checkpoint tester
- metrics (PSNR·SSIM·LPIPS·Speckle SNR·FID·KID·LNCC)
- Input | Fake | Real 3-열 콜라주  (W&B 업로드, 최대 20쌍)
"""

import os, copy, time, argparse, sys, re
from pathlib import Path
import torch, wandb
from PIL import Image
import numpy as np

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.offline_metrics import evaluate_pairwise
from util import util


# ───────────────── weight path resolver ──────────────────
def _resolve_weight_path(load_path: str) -> str:
	p = Path(load_path)
	m = re.match(r'(best_[0-9a-f]{8})_net_(G_[AB]|D_[AB])\.pth$', p.name)
	if m:
		alt = p.parent / m.group(1) / f'net_{m.group(2)}.pth'
		if alt.exists():
			return alt.as_posix()
	return load_path
# ─────────────────────────────────────────────────────────


def run_single_direction(opt, ckpt_tag, direction, device):
	"""A→B 또는 B→A 실행, 메트릭 반환 + Input/Fake/Real 콜라주 로그"""
	opt = copy.deepcopy(opt)
	opt.direction, opt.phase = direction, 'test'
	opt.serial_batches, opt.num_threads, opt.no_flip = True, 0, True
	opt.epoch = ckpt_tag

	# ─ dataset & model ─
	dataset = create_dataset(opt)

	# 빠른 검증: 첫 샘플 SRC·TGT 경로 출력
	samp = next(iter(dataset))
	src0 = samp['A_paths' if direction == 'AtoB' else 'B_paths'][0]
	tgt0 = samp['B_paths' if direction == 'AtoB' else 'A_paths'][0]
	print(f"[{direction}] SRC → {src0}\n          TGT → {tgt0}")

	model = create_model(opt)
	orig_load = torch.load
	torch.load = lambda path, **kw: orig_load(_resolve_weight_path(path), **kw)
	model.setup(opt)
	torch.load = orig_load
	model.eval()

	src_paths, real_paths, fake_paths = [], [], []
	for data_i in dataset:
		model.set_input(data_i)
		model.test()

		if direction == 'AtoB':
			src_path = data_i['A_paths'][0]          # 입력 Low
			real_path = data_i['B_paths'][0]         # 타깃 High
			fake_img = model.get_current_visuals()['fake_B']
		else:  # BtoA
			src_path = data_i['B_paths'][0]          # 입력 High
			real_path = data_i['A_paths'][0]         # 타깃 Low
			fake_img = model.get_current_visuals()['fake_A']

		fname = os.path.basename(real_path)
		save_dir = Path(opt.results_dir) / opt.name / ckpt_tag
		save_dir.mkdir(parents=True, exist_ok=True)
		fake_path = save_dir / f'{direction}_fake_{fname}'
		util.save_image(util.tensor2im(fake_img), fake_path)

		src_paths.append(src_path)
		real_paths.append(real_path)
		fake_paths.append(str(fake_path))

	# ─── W&B : Input | Fake | Real 3-열 콜라주 (≤20) ───
	if wandb.run:
		composites = []
		for s, f, r in zip(src_paths[:20], fake_paths[:20], real_paths[:20]):
			img_s = Image.open(s).convert("RGB")
			img_f = Image.open(f).convert("RGB")
			img_r = Image.open(r).convert("RGB")
			h = min(img_s.height, img_f.height, img_r.height)

			def rez(im): return im.resize((int(im.width * h / im.height), h))
			isrc, ifake, ireal = map(rez, (img_s, img_f, img_r))
			w_total = isrc.width + ifake.width + ireal.width
			combo = Image.new("RGB", (w_total, h))       # ← (w,h) 튜플
			x = 0
			for im in (isrc, ifake, ireal):
				combo.paste(im, (x, 0))
				x += im.width
			composites.append(
				wandb.Image(np.array(combo),
				            caption=f"{direction}  |  Input | Fake | Real")
			)
		wandb.log({f"{direction}/Input_Fake_Real": composites})

	return evaluate_pairwise(real_paths, fake_paths, device)


def main():
	# ─ argparse : ckpt_tag만 먼저
	cli = argparse.ArgumentParser()
	cli.add_argument('--ckpt_tag', required=True, help='best_<hash>')
	args, remaining = cli.parse_known_args()

	# ─ TestOptions 파싱
	sys.argv = [sys.argv[0]] + remaining
	opt = TestOptions().parse()
	opt.model = getattr(opt, 'model', 'cycle_gan') or 'cycle_gan'
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# ─ W&B 런
	wandb.init(
		project=getattr(opt, 'wandb_project_name', 'CycleGAN-test'),
		name=f"test_{args.ckpt_tag}",
		config={**vars(opt), "ckpt_tag": args.ckpt_tag},
	)

	t_start = time.time()
	for direction in ['AtoB', 'BtoA']:
		metrics = run_single_direction(opt, args.ckpt_tag, direction, device)
		wandb.log({f"{direction}/{k}": v for k, v in metrics.items()})
		print(f"[{direction}] " + ", ".join(f"{k}:{v:.3f}" for k, v in metrics.items()))

	print(f"✅ Done in {time.time()-t_start:.1f}s")
	wandb.finish()


if __name__ == '__main__':
	main()