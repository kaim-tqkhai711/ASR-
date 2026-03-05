import os
import sys
import subprocess
import shutil
import argparse
import json
import math
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_OUTPUT_DIR = Path("ASRmodel")
SPLIT_DURATION_SEC = 600  # 10 minutes per chunk to be safe

# --- FFmpeg Fix ---
# Add FFmpeg to PATH if not found
if not shutil.which("ffmpeg"):
    possible_ffmpeg_paths = [
        r"C:\Users\khai9\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin",
        r"C:\Users\khai9\AppData\Local\Microsoft\WinGet\Links" 
    ]
    for p in possible_ffmpeg_paths:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "ffmpeg.exe")):
            print(f"🔹 Adding {p} to PATH")
            os.environ["PATH"] += os.pathsep + p
            break
# ------------------

def get_python_script_path(cmd_name):
    """
    Finds the executable path for a python script/command (like demucs).
    """
    scripts_dir = Path(sys.prefix) / "Scripts"
    exe_path = scripts_dir / f"{cmd_name}.exe"
    if exe_path.exists(): return str(exe_path)
    script_path = scripts_dir / cmd_name
    if script_path.exists(): return str(script_path)
    path_from_shutil = shutil.which(cmd_name)
    if path_from_shutil: return path_from_shutil
    return cmd_name

def get_audio_duration(file_path):
    """
    Returns the duration of the audio file in seconds using ffprobe.
    """
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "json", 
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as e:
        print(f"⚠️ Could not determine audio duration: {e}")
        return 0

def split_audio(input_file, output_dir, segment_time):
    """
    Splits audio into chunks of `segment_time` seconds using ffmpeg.
    Returns a list of chunk file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = input_file.stem
    extension = input_file.suffix
    
    # Output pattern: basename_000.ext, basename_001.ext, ...
    output_pattern = output_dir / f"{basename}_%03d{extension}"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-c", "copy",
        str(output_pattern)
    ]
    
    print(f"   -> Splitting audio into {segment_time}s chunks...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Collect generated files
    chunks = sorted(output_dir.glob(f"{basename}_*{extension}"))
    return chunks

def concat_audio(file_list, output_file):
    """
    Concatenates a list of audio files into one using ffmpeg.
    """
    concat_list_path = output_file.parent / "concat_list.txt"
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for file_path in file_list:
            # ffmpeg requires forward slashes and safe escaping
            safe_path = str(file_path.absolute()).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_path),
        "-c", "copy",
        str(output_file)
    ]
    
    print(f"   -> Concatenating {len(file_list)} files to {output_file.name}...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup list file
    if concat_list_path.exists():
        concat_list_path.unlink()

def run_demucs_on_file(input_file, output_root):
    """
    Helper to run demucs on a single file.
    """
    demucs_exe = get_python_script_path("demucs")
    # ASRmodel/temp
    temp_dir = output_root / "temp"
    
    cmd = [
        demucs_exe, 
        "-n", "htdemucs", 
        "--two-stems=vocals", 
        str(input_file), 
        "-o", str(temp_dir)
    ]
    
    # print(f"      Running Demucs on {input_file.name}...")
    # Suppress output to keep it clean, or keep it if debugging needed
    # Using subprocess.run directly lets demucs print its progress bars
    subprocess.run(cmd, check=True)

def run_demucs(input_file, output_dir=DEFAULT_OUTPUT_DIR):
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file '{input_path}' not found.")
        return

    print(f"🔹 Input File: {input_path}")
    print(f"🔹 Output Root: {output_dir.resolve()}")

    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Target output file
    target_output = temp_dir / "htdemucs" / input_path.stem / "vocals.wav"
    if target_output.exists():
        print(f"   -> Vocals file already exists at: {target_output}")
        print("   -> Skipping Demucs run.")
        return

    print("\n[Step 1] Processing Audio with Demucs...")

    # Check duration
    duration = get_audio_duration(input_path)
    print(f"   -> Audio Duration: {duration/60:.2f} minutes")

    if duration > SPLIT_DURATION_SEC:
        print(f"   ⚠️ File is too long (> {SPLIT_DURATION_SEC/60:.0f} mins). Switching to chunked processing to avoid OOM.")
        
        # 1. Split
        chunk_dir = temp_dir / "chunks" / input_path.stem
        try:
            if chunk_dir.exists(): shutil.rmtree(chunk_dir)
        except: pass
        
        chunks = split_audio(input_path, chunk_dir, SPLIT_DURATION_SEC)
        
        processed_vocals = []
        
        # 2. Process each chunk
        for i, chunk in enumerate(chunks):
            print(f"   -> Processing chunk {i+1}/{len(chunks)}: {chunk.name}")
            try:
                run_demucs_on_file(chunk, output_dir)
                
                # Expected output for this chunk
                # Demucs output: output_dir/temp/htdemucs/{chunk_name}/vocals.wav
                chunk_vocal = temp_dir / "htdemucs" / chunk.stem / "vocals.wav"
                
                if chunk_vocal.exists():
                    processed_vocals.append(chunk_vocal)
                else:
                    print(f"❌ Error: Demucs output not found for chunk {chunk.name}")
                    return
            except Exception as e:
                print(f"❌ Error processing chunk {chunk.name}: {e}")
                return

        # 3. Concatenate
        print("   -> Merging separated vocals...")
        # Create the final directory structure manually since we tend to overwrite it
        target_output.parent.mkdir(parents=True, exist_ok=True)
        concat_audio(processed_vocals, target_output)
        
        print("✅ Chunked Demucs processing finished successfully.")
        
        # Cleanup chunks?
        # shutil.rmtree(chunk_dir) 
        
    else:
        # Standard run
        try:
            run_demucs_on_file(input_path, output_dir)
            print("✅ Demucs finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Demucs failed with error code {e.returncode}")
        except Exception as e:
            print(f"❌ An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Demucs for vocal separation.")
    parser.add_argument("input_file", nargs="?", help="Path to input audio file (wav, mp3, etc.)")
    parser.add_argument("--output_dir", default="ASRmodel", help="Root output directory")
    
    args = parser.parse_args()
    input_file = args.input_file
    
    if not input_file:
        print("🔹 No input file specified. Searching for audio files in current directory...")
        audio_extensions = {".wav", ".mp3", ".m4a", ".flac"}
        for f in Path(".").iterdir():
            if f.suffix.lower() in audio_extensions:
                input_file = f
                print(f"   -> Found: {input_file}")
                break
    
    if not input_file:
        print("❌ No audio file found or specified.")
        return

    run_demucs(input_file, Path(args.output_dir))

if __name__ == "__main__":
    main()
