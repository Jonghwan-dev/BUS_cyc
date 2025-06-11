#!/bin/bash
# 간단한 실행 스크립트
# 옵션 설명:
#   -g 사용 GPU 번호 (기본 0)
#   -k K-Fold 수 (기본 5)
#   -n 실험 이름
#   -p WandB 프로젝트 이름
#   -d 데이터 경로 (기본 ./data/USenhanced23)
#   -h 도움말 출력

GPU=0
KFOLD=5
NAME="kfold_run"
PROJECT="CycleGAN-KFold"
DATAROOT="./data/USenhanced23"

usage() {
  echo "사용법: $0 [-g GPU] [-k K] [-n NAME] [-p PROJECT] [-d DATAROOT]" >&2
}

while getopts hg:k:n:p:d: flag; do
  case "$flag" in
    g) GPU=${OPTARG} ;;
    k) KFOLD=${OPTARG} ;;
    n) NAME=${OPTARG} ;;
    p) PROJECT=${OPTARG} ;;
    d) DATAROOT=${OPTARG} ;;
    h) usage; exit 0 ;;
  esac
done

python train_kfold.py \
  --gpu_ids ${GPU} \
  --dataroot ${DATAROOT} \
  --k_folds ${KFOLD} \
  --name ${NAME} \
  --use_wandb \
  --wandb_project_name ${PROJECT}
