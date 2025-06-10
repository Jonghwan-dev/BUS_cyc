# 초음파 Super‑Resolution & 아티팩트 인지 이미지 생성 평가 지표 정리
> **대상** : 휴대용 초음파(POCUS) ↔ 고급 카트형 초음파 간 해상도·아티팩트 격차를 CycleGAN·SR 모델이 얼마나 잘 재현/개선했는지 평가  

---

## 1. Full‑Reference (정답 영상 필요)

| 지표 | 수식 (LaTeX) | 측정 요소 | 초음파 적용 이유 | 핵심 레퍼런스 |
|------|-------------|-----------|-----------------|--------------|
| **PSNR** | $$\text{PSNR}=20\log_{10}(MAX_I)-10\log_{10}(\text{MSE})$$ | 픽셀 오차 (정량) | HR 참조가 있을 때 복원 오차 파악 | [Wang & Bovik 2009](https://doi.org/10.1109/MSP.2008.930649) |
| **SSIM** | $$SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}$$ | 밝기·대조·구조 | 병변 형태·테두리 보존 확인 | [Wang *et al.* 2004](https://doi.org/10.1109/TIP.2003.819861) |
| **LPIPS** | $$\text{LPIPS}=\sum_{l} w_l\,\frac{1}{HW}\|F_l(x)-F_l(y)\|_2^2$$ | 딥 특징 유사도 (지각) | 스펙클·텍스처가 실제처럼 보이는지 | [Zhang *et al.* 2018](https://arxiv.org/abs/1801.03924) |
| **DISTS** | $$DISTS=\alpha\,D_{\text{struct}}+(1-\alpha)\,D_{\text{text}}$$ | 구조 **+** 텍스처 통합 | 가짜/과도한 스펙클 탐지 | [Ding *et al.* 2022](https://doi.org/10.1109/TPAMI.2020.3027314) |
| **FSIM** | Phase Congruency & Gradient Magnitude 기반 | 엣지·위상 정합 | 미세 경계·음영 민감 | [Zhang *et al.* 2011](https://doi.org/10.1109/TIP.2011.2105962) |

---

## 2. Distribution‑Level (세트 단위, 정답 불필요)

| 지표 | 핵심 수식 | 검증 내용 | CycleGAN 적합성 | 핵심 레퍼런스 |
|------|-----------|-----------|----------------|--------------|
| **FID** | $$\|\mu_r-\mu_g\|_2^2+\operatorname{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$$ | 실영상 vs 생성영상 분포 거리 | unpaired 전체 품질·다양성 | [Heusel *et al.* 2017](https://arxiv.org/abs/1706.08500) |
| **KID** | 다항 MMD (Inception 특징) | 작은 데이터셋에서도 안정 | HR 데이터 적을 때 보완 | [Bińkowski *et al.* 2018](https://arxiv.org/abs/1711.00815) |
| **SIFID** | FID를 이미지 단일 patch에 적용 | 국소 텍스처·그림자 현실성 | 유방 조직 이질성 평가 | [Ayan *et al.* 2020](https://arxiv.org/abs/2006.06066) |

---

## 3. Blind / No‑Reference 단일 영상 품질

| 지표 | 아이디어 | 장점 | 주의점 |
|------|----------|------|--------|
| **NIQE** | 자연 장면 통계 편차 | 참조 불필요, 빠름 | 자연 이미지 기준 → 초음파 해석 시 주의 |
| **BRISQUE** | 공간 NSS + SVM | 공개 코드, 단일 영상 | 동일 |
| **PI (Perceptual Index)** | $$0.5\,(NIQE+\text{Ma})$$ | 인간 평가와 높은 상관 (PIRM 2018) | Ma 점수 모델 필요 |

---

## 4. 초음파 아티팩트 전용 지표

| 지표 | 수식 · 정의 | 평가 대상 | 임상적 해석 |
|------|-------------|-----------|-------------|
| **Speckle SNR** | $\text{SNR}=\mu/\sigma$ (균질 ROI) | 스펙클 그레인 현실성 | $\approx1.9$ → 정상 speckle, ↑: 과도한 평활화 |
| **CNR** | $$\frac{\mu_L-\mu_B}{\sigma_B}$$ | 병변 대비 | 기본 가시성 |
| **gCNR** | $$gCNR = 1-\int \min(p_L(i),p_B(i))\,di$$ | 강도 분포 중첩 | 동적 범위 영향 제거 → 실제 검출력 |

---

## 5. CycleGAN 특화 체크리스트
1. **Cycle‑Consistency 오차** : $$L \rightarrow H \rightarrow \hat L$$ 의 MSE·SSIM  
2. **Identity Test** : HR 영상을 G<sub>L→H</sub>에 입력 ⇒ 변화 없어야 함  
3. **Downstream 성능** : 생성 HR 영상으로 분류·세그 테스트 (AUC, Dice 등)

---

## 참고문헌 (BibTeX)

```bibtex
@article{Wang2009MSE,
  title={Mean squared error: Love it or leave it?},
  author={Wang, Z. and Bovik, A.},
  journal={IEEE Signal Processing Magazine},
  year={2009},
  doi={10.1109/MSP.2008.930649}}

@article{Wang2004SSIM,
  title={Image quality assessment: from error visibility to structural similarity},
  author={Wang, Z. and Bovik, A. and Sheikh, H. and Simoncelli, E.},
  journal={IEEE TIP},
  year={2004},
  doi={10.1109/TIP.2003.819861}}

@inproceedings{Zhang2018LPIPS,
  title={The unreasonable effectiveness of deep features as a perceptual metric},
  author={Zhang, R. and Isola, P. and Efros, A. A.},
  booktitle={CVPR},
  year={2018},
  url={https://arxiv.org/abs/1801.03924}}

@article{Ding2022DISTS,
  title={Image Quality Assessment: Unifying Structure and Texture Similarity},
  author={Ding, K. and Ma, K. and Wang, S. and Simoncelli, E.},
  journal={IEEE TPAMI},
  year={2022},
  doi={10.1109/TPAMI.2020.3027314}}

@article{Zhang2011FSIM,
  title={FSIM: A Feature Similarity Index for Image Quality Assessment},
  author={Zhang, L. and Zhang, L. and Mou, X. and Zhang, D.},
  journal={IEEE TIP},
  year={2011},
  doi={10.1109/TIP.2011.2105962}}

@inproceedings{Heusel2017FID,
  title={GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium},
  author={Heusel, M. and Ramsauer, H. and Unterthiner, T. and Nessler, B. and Hochreiter, S.},
  booktitle={NIPS},
  year={2017},
  url={https://arxiv.org/abs/1706.08500}}

@inproceedings{Binkowski2018KID,
  title={Demystifying MMD GANs},
  author={Bińkowski, M. and Sutherland, D. J. and Arbel, M. and Gretton, A.},
  booktitle={ICLR},
  year={2018},
  url={https://arxiv.org/abs/1711.00815}}

@article{Mittal2013NIQE,
  title={Making a completely blind image quality analyzer},
  author={Mittal, A. and Soundararajan, R. and Bovik, A.},
  journal={IEEE SPL},
  year={2013},
  doi={10.1109/LSP.2012.2227726}}

@article{RodriguezMolares2020gCNR,
  title={The Generalized Contrast-to-Noise Ratio: A Formal Definition for Lesion Detectability},
  author={Rodríguez-Molares, A. and Rindal, O. M. H. and D'hooge, J. and et al.},
  journal={IEEE TUFFC},
  year={2020},
  doi={10.1109/TUFFC.2019.2956855}}
```

---

### 사용 TIP  
- **PSNR·SSIM** : 시뮬레이션 HR/LR 쌍에서 기본 확인  
- **LPIPS·DISTS** : 스펙클·음영 질감이 실제와 유사한지  
- **FID·KID** : unpaired 전역 품질 검증  
- **Speckle SNR·gCNR** : 초음파 고유 아티팩트·병변 가시성 체크  

> 모든 지표를 **조합**해 보고·학습·임상 검증 단계별로 선별 적용