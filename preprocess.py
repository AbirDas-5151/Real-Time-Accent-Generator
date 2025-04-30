import os
import librosa
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

SOURCE_DIR = "data"
TARGET_DIR = "data_preprocessed"
TARGET_SAMPLE_RATE = 16000

def convert_audio(file_path, target_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Resample to 16kHz
    if sr != TARGET_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

    # Normalize volume
    y = y / max(abs(y))

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # Save as WAV
    sf.write(target_path, y, TARGET_SAMPLE_RATE)

def process_folder(source_subfolder):
    source_path = os.path.join(SOURCE_DIR, source_subfolder)
    target_path = os.path.join(TARGET_DIR, source_subfolder)
    os.makedirs(target_path, exist_ok=True)

    for file in tqdm(os.listdir(source_path), desc=f"Processing {source_subfolder}"):
        if file.endswith(".wav"):
            src_file = os.path.join(source_path, file)
            tgt_file = os.path.join(target_path, file)
            convert_audio(src_file, tgt_file)

if __name__ == "__main__":
    for folder in ["my_voice", "us_accent"]:
        process_folder(folder)
