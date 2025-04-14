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
    INSTALL_CMD="apt-get install -y ffmpeg git cmake build-essential"
else
    UPDATE_CMD="sudo apt-get update"
    INSTALL_CMD="sudo apt-get install -y ffmpeg git cmake build-essential"
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
install_package "awscli"

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
        aws s3 cp s3://muse-gs-123/resources/ggml-large-v3-turbo.bin ../resources/
        aws s3 cp s3://muse-gs-123/resources/ggml-large-v3-turbo-encoder.mlmodelc/ ../resources/ --recursive
    fi
    
    # 항상 기본 빌드 방식으로 진행 (CoreML 관련 옵션 제거)
    cmake -B build
    cmake --build build --config Release -j
    cd ..
else
    echo "whisper.cpp가 이미 존재합니다."
fi

echo "========== Silero VAD 모델 사전 다운로드 =========="
python -c "import torch; torch.hub.load('snakers4/silero-vad', model='silero_vad')" || echo "Silero VAD 모델 다운로드 실패, 나중에 재시도됩니다."

echo "========== 메인 스크립트 실행 =========="
# 여기에서는 CLI 모드로만 실행되도록 설정
echo "시스템 및 Whisper 설치가 완료되었습니다."
echo "오디오 필사를 시작하려면, 다음과 같이 실행하세요:"
echo "  python STT_Voice_Spliter.py /경로/파일.wav"