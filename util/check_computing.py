import os
import platform
import psutil
import subprocess
import torch

def get_cpu_info():
    print(" CPU 정보 ----------------------------")
    print(f" - 논리적 코어 수 : {psutil.cpu_count(logical=True)}")
    print(f" - 물리적 코어 수 : {psutil.cpu_count(logical=False)}")
    print(f" - CPU 사용률     : {psutil.cpu_percent(interval=1)}%")
    print(f" - CPU 모델       : {platform.processor()}")
    print()

def get_memory_info():
    print(" 메모리 정보 -------------------------")
    mem = psutil.virtual_memory()
    print(f" - 총 메모리     : {round(mem.total / 1024**3, 2)} GiB")
    print(f" - 사용 중 메모리 : {round(mem.used / 1024**3, 2)} GiB")
    print(f" - 사용률         : {mem.percent}%")
    print()

def get_gpu_info():
    print(" GPU 정보 ----------------------------")
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,index,memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits']
        )
        gpus = output.decode().strip().split('\n')
        for gpu in gpus:
            name, index, total, used, free = gpu.split(', ')
            print(f" - GPU {index}: {name}")
            print(f"    • 메모리 총량 : {total} MiB")
            print(f"    • 사용 중     : {used} MiB")
            print(f"    • 남은 용량   : {free} MiB")
            print("PyTorch version:", torch.__version__)
            print("CUDA available:", torch.cuda.is_available())
            print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    except Exception as e:
        print(" - NVIDIA GPU 없음 또는 nvidia-smi 미설치")
        print(f"   오류: {str(e)}")
    print()

def main():
    print("시스템 환경 체크 시작 ====================================")
    get_cpu_info()
    get_memory_info()
    get_gpu_info()
    print("시스템 환경 체크 완료 ====================================")

if __name__ == "__main__":
    main()
