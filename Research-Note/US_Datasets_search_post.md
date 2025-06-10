# Ultrasound HR/LR pair datasets

   
### Research 활용 datasets link & reference 정리  
[📄 데이터셋 정리본 보기 (PDF)](file:///home/used/workspace/Public%20Ultrasound%20Datasets%20for%20Super-Resolution%20Training.pdf)  

---

## 요약 정리 표

| # | 데이터셋 | 샘플 수 · 해상도 | 장비(휴대형 vs. 카트형) | 라이선스 | SR 활용 포인트 |
|---|----------|-----------------|------------------------|----------|----------------|
| 1 | **USenhance 2023 Challenge** | 1,500 쌍 (총 3,000 장) · B‑mode | 동일 부위 재촬영: 휴대형 POCUS ↔ 고급 카트형 | 연구용 공개 (Grand‑Challenge) | *실제* HR/LR 페어를 바로 사용 |
| 2 | **BUSI** (2020) | 780 장 · 평균 500×500 px | GE LOGIQ E9 (카트형) | CC BY 4.0 | HR(원본) → 다운샘플 LR 생성 |
| 3 | **BUSIS** (2022) | 562 장 · 다중 해상도 | 5개 카트형 장비 (세대·제조사 다양) | CC BY 4.0 | 기기 간 화질 편차 또는 인위적 LR |
| 4 | **BUS‑UCLM** (2025) | 683 장 · 최대 ≈ 1024×768 px | 최신 카트형 (Siemens Acuson S2000) | CC BY 4.0 | 고품질 HR로 사용, 다운샘플 LR 생성 |
| 5 | **OASBUD** (2017) | 100 병변 원시 RF 신호 | 카트형 (모델 미상) | CC BY‑NC 4.0 | 원시 RF로 HR 재구성 → 빔 수 감소 등 물리 기반 LR |
| 6 | **COVIDx‑US** (2022, Lung) | 242 비디오 ≈ 29k 프레임 | 전부 휴대형 POCUS | 오픈 액세스 | 실제 저화질 POCUS로 도메인 일반화 |

## 활용 팁
- **직접 페어 제공**: USenhance 2023이 유일하게 완전 HR/LR 페어를 제공하므로 SR 네트워크 기본 학습용으로 최적.
- **가상 페어 생성**: BUSI·BUSIS·BUS‑UCLM은 원본을 HR로 두고, 다운샘플·노이즈 삽입 등으로 LR을 만들어 학습/평가.
- **물리 기반 저해상도**: OASBUD의 원시 RF를 활용하면 실제 하드웨어 제약(빔 수, 주파수 등)을 모사한 LR 생성이 가능.
- **도메인 일반화**: COVIDx‑US는 휴대형 POCUS 특유의 노이즈·아티팩트 학습에 사용해, 휴대형 입력에도 견고한 모델 구현.

---

## 간단 BibTeX 참고

```bibtex
@misc{Guo2023USEnhance,
  title  = {Ultrasound Image Enhancement Challenge 2023 Dataset},
  author = {Guo, Y. et al.},
  year   = {2023},
  note   = {1,500 paired portable vs. cart ultrasound images}
}
@article{AlDhabyani2020BUSI,
  title  = {Dataset of breast ultrasound images},
  author = {Al-Dhabyani, W. et al.},
  journal= {Data in Brief},
  year   = {2020}
}
@article{Cheng2022BUSIS,
  title  = {BUSIS: A Benchmark for Breast Ultrasound Image Segmentation},
  author = {Cheng, H.-D. et al.},
  year   = {2022}
}
@article{Vallez2025BUSUCLM,
  title  = {BUS-UCLM: Breast ultrasound lesion segmentation dataset},
  author = {Vallez, N. et al.},
  journal= {Scientific Data},
  year   = {2025}
}
@article{Piotrzkowska2017OASBUD,
  title  = {Open-access raw breast ultrasound signals},
  author = {Piotrzkowska-Wróblewska, H. et al.},
  journal= {Medical Physics},
  year   = {2017}
}
@article{Ebadi2022COVIDxUS,
  title  = {COVIDx-US: Ultrasound benchmark for COVID-19},
  author = {Ebadi, A. et al.},
  journal= {Frontiers in Bioinformatics},
  year   = {2022}
}
```