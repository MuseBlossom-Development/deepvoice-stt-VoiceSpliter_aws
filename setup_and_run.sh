#!/bin/bash
set -e  # 오류 발생 시 스크립트 종료

echo "========== Miniconda 확인 =========="

if ! command -v conda &> /dev/null; then
    echo "Miniconda가 설치되어 있지 않습니다. Miniconda를 설치합니다..."

    OS_TYPE=$(uname)
    ARCH_TYPE=$(uname -m)

    if [[ "$OS_TYPE" == "Linux" ]]; then
        # Linux의 경우
        if [[ "$ARCH_TYPE" == "aarch64" ]]; then
            MINICONDA_SCRIPT="Miniconda3-latest-Linux-aarch64.sh"
        else
            MINICONDA_SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
        fi
    else
        echo "지원되지 않는 운영체제: $OS_TYPE"
        exit 1
    fi

    echo "설치 스크립트: $MINICONDA_SCRIPT"
    curl -L https://repo.anaconda.com/miniconda/$MINICONDA_SCRIPT -o $MINICONDA_SCRIPT
    
    # 설치 실행
    bash $MINICONDA_SCRIPT -u -b -p $HOME/miniconda3
    
    # 환경 변수 설정
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    # conda 초기화
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    
    # 설치 스크립트 삭제
    rm $MINICONDA_SCRIPT
else
    echo "Miniconda가 이미 설치되어 있습니다."
    # conda 활성화
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "========== Conda 환경 생성 =========="
# 기존의 stt_env 환경이 있으면 제거
if conda env list | grep -q "stt_env"; then
    echo "기존 stt_env 환경을 제거합니다..."
    conda deactivate || true
    conda env remove -n stt_env --yes || echo "환경 제거 실패, 계속 진행합니다."
    if [ -d "$(conda info --base)/envs/stt_env" ]; then
        echo "환경 폴더가 남아있어 수동으로 제거합니다..."
        rm -rf "$(conda info --base)/envs/stt_env"
    fi
fi

echo "새로운 stt_env 환경을 생성합니다..."
conda create -y -n stt_env python=3.11 tk ffmpeg 

echo "========== Conda 환경 활성화 =========="
conda activate stt_env || { echo "환경 활성화 실패"; exit 1; }
echo "현재 활성화된 환경: $(conda info --envs | grep '*')"

echo "========== 시스템 종속성 설치 =========="
# Linux (Ubuntu/Debian 기준)에서 root 여부에 따라 sudo 사용 여부 결정
echo "필요한 시스템 패키지 설치 중..."
if [ "$EUID" -eq 0 ]; then
    UPDATE_CMD="apt-get update"
    INSTALL_CMD="apt-get install -y ffmpeg git cmake build-essential libavcodec-dev libavformat-dev libavutil-dev"
else
    UPDATE_CMD="sudo apt-get update"
    INSTALL_CMD="sudo apt-get install -y ffmpeg git cmake build-essential libavcodec-dev libavformat-dev libavutil-dev"
fi

$UPDATE_CMD
$INSTALL_CMD

echo "========== Python 패키지 설치 =========="
install_package() {
    echo "패키지 설치: $1"
    pip install "$1" || echo "경고: $1 설치 실패, 계속 진행합니다."
}

# 기본 패키지
install_package "requests"
install_package "aiohttp" 
install_package "aiofiles"
install_package "numpy"
# awscli 관련 코드 제거: install_package "awscli"

# Linux용 PyTorch 설치
install_package "torch"
install_package "torchaudio"

echo "========== whisper.cpp 다운로드 및 빌드 =========="
if [ ! -d "whisper.cpp" ]; then
    git clone https://github.com/ggml-org/whisper.cpp.git
    cd whisper.cpp
    
    # 필요한 디렉토리 생성
    mkdir -p models
    
    # resources 폴더에서 모델 및 엔코더 파일 복사 (이미 존재한다고 가정)
    if [ -d "../resources" ]; then
        echo "resources 폴더에서 모델 파일 복사 중..."
        if [ -f "../resources/ggml-large-v3-turbo.bin" ]; then
            cp "../resources/ggml-large-v3-turbo.bin" models/
        fi
        
        if [ -d "../resources/ggml-large-v3-turbo-encoder.mlmodelc" ]; then
            cp -r "../resources/ggml-large-v3-turbo-encoder.mlmodelc" models/
        fi
    else
        echo "resources 폴더가 없습니다. 필요한 모델 파일이 있는지 확인하세요."
        aws s3 cp --no-sign-request s3://muse-gs-123/resources ../resources/ --recursive
    fi
    
    # GPU 지원 옵션 결정
    if command -v nvidia-smi &> /dev/null; then
        GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
        echo "GPU Compute Capability: $GPU_CC"
        # 비교를 위해 bc 사용 (8.6 이상이면 최신 GPU로 간주)
        if (( $(echo "$GPU_CC >= 8.6" | bc -l) )); then
            echo "신형 NVIDIA GPU가 감지되었습니다. CUDA 지원 빌드를 진행합니다. (-DCMAKE_CUDA_ARCHITECTURES=\"86\")"
            cmake -B build -DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="86"
        else
            echo "구형 NVIDIA GPU가 감지되었습니다. 기본 CUDA 빌드를 진행합니다."
            # cmake -B build -DGGML_CUDA=1
            cmake -B build 
        fi
    else
        echo "NVIDIA GPU가 감지되지 않았습니다. 기본 빌드를 진행합니다."
        cmake -B build
    fi
    cmake --build build --config Release -j
    cd ..
else
    echo "whisper.cpp가 이미 존재합니다."
fi

echo "========== Silero VAD 모델 사전 다운로드 =========="
python -c "import torch; torch.hub.load('snakers4/silero-vad', model='silero_vad')" || echo "Silero VAD 모델 다운로드 실패, 나중에 재시도됩니다."

echo "========== 메인 스크립트 실행 =========="
MARKER="installed.flag"
if [ ! -f "$MARKER" ]; then
    echo "첫 설치로 판단됩니다. 같은 폴더에 있는 Tvoice.mp3 파일을 이용하여 자동 테스트를 진행합니다."
    TEST_FILE="$(pwd)/Tvoice.mp3"
    if [ -f "$TEST_FILE" ]; then
        START_TIME=$(date +%s)
        python STT_Voice_Spliter.py "$TEST_FILE"
        END_TIME=$(date +%s)
        ELAPSED_TIME=$((END_TIME - START_TIME))
        echo "전체 프로세싱 시간: ${ELAPSED_TIME}초"
        # 테스트 파일에 기반하여 생성된 폴더 제거
        TEST_BASENAME=$(basename "$TEST_FILE" .mp3)
        TEST_FOLDER="split_audio/$TEST_BASENAME"
        if [ -d "$TEST_FOLDER" ]; then
            echo "테스트 폴더($TEST_FOLDER)를 삭제합니다."
            rm -rf "$TEST_FOLDER"
        else
            echo "테스트 폴더 $TEST_FOLDER가 존재하지 않습니다."
        fi
        touch "$MARKER"
        echo "자동 테스트 완료 및 설치 마커 생성: $MARKER"
    else
        echo "Tvoice.mp3 파일이 현재 폴더에 존재하지 않습니다. Tvoice.mp3 파일을 복사한 후 다시 실행하세요."
        exit 1
    fi
else
    echo "설치가 이미 완료되었습니다. 오디오 필사를 시작하려면 파일 경로를 입력하세요."
    read -p "필사할 오디오 파일의 경로: " INPUT_FILE
    if [ -f "$INPUT_FILE" ]; then
        python STT_Voice_Spliter.py "$INPUT_FILE"
    else
        echo "입력한 파일이 존재하지 않습니다."
        exit 1
    fi
fi