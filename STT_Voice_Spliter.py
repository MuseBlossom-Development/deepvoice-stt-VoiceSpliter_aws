#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import shutil
import json
import argparse
import time
import zipfile
import asyncio
import threading
import queue
import torch, torchaudio
import requests
import aiohttp, aiofiles

# ============================================
# PyInstaller 번들 리소스 경로 추출 함수
def resource_path(relative_path):
    """
    PyInstaller 번들링된 실행 파일 내에서 리소스 파일의 절대 경로를 반환합니다.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ============================================
# 외부 바이너리 경로 설정
ffmpeg_path = "ffmpeg"
git_path    = "git"
cmake_path  = "cmake"

env = os.environ.copy()
env["PATH"] = f"{resource_path('.')}{os.pathsep}" + env.get("PATH", "")

# ============================================
# 시스템 의존성 체크 및 설치 함수들
def is_installed(command):
    return shutil.which(command) is not None

def install_with_brew(package):
    try:
        print(f"{package} 설치 시작...")
        subprocess.run(["brew", "install", package], check=True)
        print(f"{package} 설치 완료.")
    except subprocess.CalledProcessError as e:
        print(f"{package} 설치 중 오류: {e}")
        sys.exit(1)

def check_and_install_system_dependencies(progress_callback=lambda msg, val: print(msg)):
    if platform.system() == "Linux":
        progress_callback("필요한 시스템 패키지 설치 중...", 0)
        if os.geteuid() == 0:
            update_cmd = "apt-get update"
            install_cmd = "apt-get install -y ffmpeg git cmake build-essential"
        else:
            update_cmd = "sudo apt-get update"
            install_cmd = "sudo apt-get install -y ffmpeg git cmake build-essential"
        subprocess.run(update_cmd, shell=True, check=True)
        subprocess.run(install_cmd, shell=True, check=True)
    else:
        print("지원되지 않는 운영체제. ffmpeg, git, cmake를 수동으로 설치해주세요.")
        sys.exit(1)

# ============================================
# 전역 변수 및 설정
install_log_queue = queue.Queue()

CONFIG_FILE    = "config.json"
DEFAULT_CONFIG = {
    "min_speech_duration_ms": 500,
    "min_silence_duration_ms": 700,
    "max_speech_duration_s": 18,
    "speech_pad_ms": 10,
    "threshold": 0.6
}
vad_config = {}

def load_config():
    global vad_config
    config_path = resource_path(CONFIG_FILE)
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                vad_config = json.load(f)
        except Exception as e:
            print("설정 파일 로드 에러:", e)
            vad_config = DEFAULT_CONFIG.copy()
    else:
        vad_config = DEFAULT_CONFIG.copy()
    print("현재 VAD 설정:", vad_config)

def save_config():
    with open(resource_path(CONFIG_FILE), "w") as f:
        json.dump(vad_config, f, indent=4)

load_config()

# ============================================
# Whisper CLI 및 모델 경로 (PyInstaller 번들 고려)
WHISPER_CLI   = resource_path("whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = resource_path("whisper.cpp/models/ggml-large-v3-turbo.bin")

# ============================================
# 파일 크기 확인 및 비동기 다운로드 함수
def get_file_size(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
    return total

async def download_with_progress_aiohttp(url, filename, progress_callback=None):
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/109.0.0.0 Safari/537.36"),
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 65536  # 64KB
            async with aiofiles.open(filename, "wb") as f:
                async for chunk in response.content.iter_chunked(chunk_size):
                    if chunk:
                        await f.write(chunk)
                        downloaded += len(chunk)
                        if total and progress_callback:
                            percent = downloaded * 100 / total
                            progress_callback(f"다운로드 진행률: {percent:.1f}%", min(100, percent))

# ============================================
# Whisper 설치 및 빌드 관련 함수들
def check_whisper_cli():
    return os.path.exists(WHISPER_CLI)

def copy_prebundled_files(progress_callback):
    models_dir = resource_path("whisper.cpp/models")
    os.makedirs(models_dir, exist_ok=True)
    encoder_src = resource_path("resources/ggml-large-v3-turbo-encoder.mlmodelc")
    model_src   = resource_path("resources/ggml-large-v3-turbo.bin")
    encoder_dest = os.path.join(models_dir, "ggml-large-v3-turbo-encoder.mlmodelc")
    model_dest   = os.path.join(models_dir, "ggml-large-v3-turbo.bin")
    
    if os.path.isdir(encoder_src):
        if not os.path.exists(encoder_dest):
            shutil.copytree(encoder_src, encoder_dest)
            progress_callback("미리 번들된 엔코더 디렉토리 복사 완료", 30)
        else:
            progress_callback("엔코더 디렉토리가 이미 존재합니다.", 30)
    else:
        if not os.path.exists(encoder_dest):
            shutil.copy2(encoder_src, encoder_dest)
            progress_callback("미리 번들된 엔코더 파일 복사 완료", 30)
        else:
            progress_callback("엔코더 파일이 이미 존재합니다.", 30)
            
    if os.path.isdir(model_src):
        if not os.path.exists(model_dest):
            shutil.copytree(model_src, model_dest)
            progress_callback("미리 번들된 모델 디렉토리 복사 완료", 30)
        else:
            progress_callback("모델 디렉토리가 이미 존재합니다.", 30)
    else:
        if not os.path.exists(model_dest):
            shutil.copy2(model_src, model_dest)
            progress_callback("미리 번들된 모델 파일 복사 완료", 30)
        else:
            progress_callback("모델 파일이 이미 존재합니다.", 30)

def download_and_build_whisper(progress_callback):
    target_dir = resource_path("whisper.cpp")
    if not os.path.exists(target_dir):
        progress_callback("Whisper.cpp 저장소 클론 중...", 10)
        try:
            subprocess.run([git_path, "clone", "https://github.com/ggml-org/whisper.cpp.git"],
                           check=True, env=env)
        except subprocess.CalledProcessError as e:
            progress_callback(f"Git 클론 실패: {e}", 10)
            progress_callback("미리 번들된 파일을 사용합니다.", 10)
            copy_prebundled_files(progress_callback)
    else:
        progress_callback("Whisper.cpp 저장소가 이미 존재합니다.", 10)
        
    copy_prebundled_files(progress_callback)
    
    progress_callback("CMake 구성 중...", 40)
    subprocess.run(["cmake", "-B", "build"],
                   cwd=resource_path("whisper.cpp"), check=True, env=env)
    progress_callback("프로젝트 빌드 중...", 50)
    if platform.system() == "Windows":
        build_command = ["cmake", "--build", "build", "--config", "Release"]
    else:
        build_command = ["cmake", "--build", "build"]
    subprocess.run(build_command, cwd=resource_path("whisper.cpp"), check=True, env=env)
    progress_callback("Whisper.cpp 빌드 완료", 70)

def installation_process(progress_callback):
    try:
        progress_callback("시스템 환경 확인 중...", 0)
        os_type = platform.system()
        arch = platform.machine()
        progress_callback(f"운영체제: {os_type}, 아키텍처: {arch}", 5)
        check_and_install_system_dependencies(progress_callback)
        if not check_whisper_cli():
            progress_callback("Whisper CLI가 없습니다. 다운로드 및 빌드 진행...", 5)
            download_and_build_whisper(progress_callback)
        else:
            progress_callback("Whisper CLI가 이미 존재합니다.", 70)
        progress_callback("전체 설치 완료", 100)
        return True
    except Exception as e:
        progress_callback(f"오류 발생: {e}", 100)
        raise

# ============================================
# 오디오 필사 관련 함수들
def check_ffmpeg():
    if os.system(f'"{ffmpeg_path}" -version') != 0:
        print("❌ FFmpeg가 설치되어 있지 않습니다.")
        sys.exit(1)

def convert_to_mp3(file_path):
    if file_path.lower().endswith(".mp3"):
        print(f"🎵 이미 MP3: {file_path}")
        return file_path
    output_mp3 = file_path.rsplit(".", 1)[0] + ".mp3"
    print(f"🔄 WAV → MP3: {file_path} → {output_mp3}")
    command = f'"{ffmpeg_path}" -i "{file_path}" -c:a libmp3lame -b:a 128k "{output_mp3}" -y'
    os.system(command)
    if not os.path.exists(output_mp3):
        print(f"❌ MP3 변환 실패: {output_mp3}")
        sys.exit(1)
    return output_mp3

def split_audio(file_path):
    check_ffmpeg()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = os.path.join("split_audio", base_name)
    mp3_folder = os.path.join(output_folder, "MP3")
    text_folder = os.path.join(output_folder, "TEXT")
    os.makedirs(mp3_folder, exist_ok=True)
    os.makedirs(text_folder, exist_ok=True)
    file_path = convert_to_mp3(file_path)
    print("🔍 Silero VAD 모델 로딩 중...")
    model, utils = torch.hub.load('snakers4/silero-vad', model='silero_vad')
    get_speech_timestamps = utils[0]
    read_audio = utils[2]
    print(f"🎵 오디오 로드: {file_path}")
    wav = read_audio(file_path, sampling_rate=16000)
    print("🧠 음성 구간 감지 중...")
    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=16000,
        min_speech_duration_ms=vad_config.get("min_speech_duration_ms", 500),
        min_silence_duration_ms=vad_config.get("min_silence_duration_ms", 700),
        max_speech_duration_s=vad_config.get("max_speech_duration_s", 18),
        speech_pad_ms=vad_config.get("speech_pad_ms", 10),
        threshold=vad_config.get("threshold", 0.6)
    )
    total_segments = len(speech_timestamps)
    print(f"✅ 감지 구간: {total_segments} 개")
    for idx, segment in enumerate(speech_timestamps):
        start_time = max(0, segment['start'] / 16000)
        end_time = segment['end'] / 16000
        duration = end_time - start_time
        if start_time >= end_time or duration < 0.5:
            print(f"⚠️ 스킵됨: {idx+1}.mp3 (잘못된 구간)")
            continue
        output_mp3 = os.path.join(mp3_folder, f"{idx+1}.mp3")
        command = f'"{ffmpeg_path}" -i "{file_path}" -ss {start_time} -t {duration} -c copy "{output_mp3}" -y'
        print(f"🔪 MP3 분할: {idx+1}/{total_segments}")
        os.system(command)
    print("✅ 분할 완료!")
    return output_folder, mp3_folder, text_folder

def remove_newlines_from_text(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    cleaned_text = " ".join(line.strip() for line in lines)
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"✅ 개행 제거: {text_file}")

def transcribe_audio(mp3_folder, text_folder):
    if not os.path.exists(mp3_folder):
        print(f"❌ 폴더 없음: {mp3_folder}")
        sys.exit(1)
    audio_files = sorted([os.path.join(mp3_folder, f) for f in os.listdir(mp3_folder) if f.endswith(".mp3")])
    total_files = len(audio_files)
    print(f"🎤 Whisper 필사: {total_files}개 파일")
    whisper_command = [
        WHISPER_CLI,
        "--model", WHISPER_MODEL,
        "--language", "ko",
        "--output-txt"
    ] + audio_files
    print("🚀 Whisper CLI 실행 중...")
    proc = subprocess.Popen(whisper_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(proc.stdout.readline, ""):
        print(line.strip())
    proc.wait()
    for file in audio_files:
        txt_file = f"{file}.txt"
        if os.path.exists(txt_file):
            remove_newlines_from_text(txt_file)
            new_location = os.path.join(text_folder, os.path.basename(txt_file))
            shutil.move(txt_file, new_location)
            print(f"✅ {os.path.basename(file)} → {new_location}")
        else:
            print(f"❌ TXT 파일 생성 실패: {txt_file}")
    print(f"🎉 필사 완료! 결과: {text_folder}")

# ============================================
# 메인 실행 함수 (첫 실행 시 자동 설치 및 Tvoice.mp3 필사, 이후는 입력받음)
def main():
    marker = "installed.flag"
    
    parser = argparse.ArgumentParser(description="STT Voice Splitter")
    # 파일 경로를 선택적 위치 인수로 받음
    parser.add_argument("filepath", nargs="?", help="필사를 위한 오디오 파일 경로")
    args = parser.parse_args()

    if args.filepath:
        file_path = args.filepath
    else:
        if not os.path.exists(marker):
            # 첫 실행인 경우 설치 및 기본 테스트 진행
            print("첫 실행: 시스템 의존성 및 Whisper 설치를 진행합니다.")
            installation_process(lambda msg, val: print(f"{msg} ({val}%)"))
            with open(marker, "w") as f:
                f.write("installed")
            # 같은 폴더의 Tvoice.mp3를 기본 파일로 사용
            file_path = os.path.join(os.getcwd(), "Tvoice.mp3")
            if not os.path.exists(file_path):
                print("Tvoice.mp3 파일이 존재하지 않습니다. 해당 파일을 현재 폴더에 복사해주세요.")
                sys.exit(1)
        else:
            # 이후 실행: 파일 경로를 입력받음
            file_path = input("필사할 오디오 파일의 경로를 입력하세요: ").strip()
            if not os.path.exists(file_path):
                print("입력한 파일이 존재하지 않습니다.")
                sys.exit(1)

    print("오디오 파일 처리 시작...")
    file_path = convert_to_mp3(file_path)
    output_folder, mp3_folder, text_folder = split_audio(file_path)
    transcribe_audio(mp3_folder, text_folder)
    print("오디오 처리 및 필사 완료.")

if __name__ == "__main__":
    main()