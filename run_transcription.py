import os
import sys
import json
import re
import argparse
import shutil
import warnings
from pathlib import Path
import numpy as np
import torch
import torchaudio
from datetime import datetime

# --- Numpy 2.0 Hotfix ---
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
if not hasattr(np, 'float'):
    np.float = float
# ------------------------

os.environ["OMP_NUM_THREADS"] = "1"

# ==========================================
# CONFIGURATION
# ==========================================
DEFAULT_OUTPUT_DIR = Path("ASRmodel")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_text_content(text):
    text = re.sub(r'[^\w\s]', '', text) 
    return text.strip()

def is_vietnamese_word(word_text):
    """
    Heuristic to determine if a word is Vietnamese.
    """
    chars_to_strip = '.,?!:;«»""\'\'()[]{}-'
    clean_word = word_text.strip(chars_to_strip)
    if not clean_word: return False
    w_lower = clean_word.lower()
    
    if not clean_word.isascii(): return True
    
    invalid_endings = ('f', 'j', 'l', 'r', 's', 'v', 'z', 'w', 'k', 'd', 'b') 
    if w_lower.endswith(invalid_endings): return False
        
    en_blacklist = {
        "ok", "okay", "video", "clip", "view", "like", "comment", "share", 
        "subscribe", "channel", "fan", "anti", "game", "show", "live",
        "is", "am", "are", "does", "did", "have", "has", "had", 
        "but", "and", "or", "for", "in", "at", "of", "by", "my", "me", "he", "she", "it", "we", "they",
        "go", "see", "look", "watch", "eat", "drink", "rice", "noodle", "food"
    }
    if w_lower in en_blacklist: return False

    vi_common_syllables = {
        "anh", "em", "chi", "co", "chu", "bac", "ong", "ba", "me", "bo",
        "toi", "ban", "cau", "to", "minh", "no", "ho", "chung", "ta", 
        "la", "ma", "va", "thi", "la", "o", "da", "dang", "se", "chua", "roi",
        "di", "ve", "an", "uong", "ngu", "nghi", "lam",
        "nhieu", "it", "lon", "nho", "to", "be", "dep", "xau",
        "trong", "ngoai", "tren", "duoi", "truoc", "sau",
        "ngay", "dem", "sang", "trua", "chieu", "toi", "nam", "thang", "tuan",
        "mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin", "muoi",
        "tram", "nghin", "trieu", "ty",
        "cai", "con", "nguoi", "xe", "nha", "cua",
        "a", "u", "uh", "uk", "ukm", "nha", "ne", "he", "hi", "ha", "haha", "hihi",
        "ok", "oke", 
        "linh", "trang", "nam", "huy", "duy", "lan", "mai", "hoa", "hung", "cuong", "dung",
        "tuan", "thanh", "thao", "phuong", "bich", "thu", "thuy", "van",
        "tra", "tranh", "banh", "manh", "im", "om", "am", "em", "in", "on", "un",
        "canh", "xanh", "lanh", "manh", "ranh", "thanh", "danh",
        "cam", "tam", "lam", "mam", "nam", "sam",
        "chan", "than", "van", "nhan", "dan", "gan", "khan",
        "cho", "lo", "mo", "bo", "do", "co", "go", "xo", "no",
        "ca", "da", "ga", "ha", "la", "ma", "na", "pa", "qa", "sa", "ta", "va", "xa",
        "bi", "chi", "di", "ghi", "hi", "ki", "li", "mi", "ni", "pi", "qi", "ri", "si", "ti", "vi", "xi",
        "bu", "cu", "du", "gu", "hu", "lu", "mu", "nu", "pu", "ru", "su", "tu", "vu", "xu",
        "kia", "nay", "no", "day", "vay", "xong",
        "ai", "cao", "sao", "tia", "rau", "bao", "so", "xay", "sai", "cay", "cai",
        "keo", "geo", "reo", "neo", "tieu", "dieu", "bieu", "mieu", "nieu", "rieu", "sieu", "yeu"
    }
    
    if w_lower in vi_common_syllables:
        return True
        
    valid_starts = ('ng', 'nh', 'th', 'tr', 'ch', 'ph', 'kh', 'gh', 'gi', 'qu')
    if w_lower.startswith(valid_starts):
        if not w_lower.endswith(invalid_endings):
            return True

    return False

# ==========================================
# MAIN LOGIC
# ==========================================
def run_transcription(input_name, output_dir=DEFAULT_OUTPUT_DIR):
    """
    input_name: essentially the stem of the original file (e.g. 'input1').
    We look for ASRmodel/temp/htdemucs/{input_name}/vocals.wav
    """
    
    # 1. Locate Vocals File
    temp_dir = output_dir / "temp"
    vocals_path = temp_dir / "htdemucs" / input_name / "vocals.wav"
    
    if not vocals_path.exists():
        print(f"❌ Error: Vocals file not found at {vocals_path}")
        print("   -> Tip: Run 'run_demucs.py' first.")
        return

    print(f"🔹 Found Vocals: {vocals_path}")
    
    # Create final run output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / input_name / f"run_{timestamp}"
    segment_out_dir = run_dir / "output_segment"
    segment_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔹 Output Directory: {run_dir}")

    # 2. WhisperX Transcribe & Align
    print("\n[Step 2] Transcribing with WhisperX...")
    import whisperx
    import gc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"   -> Device: {device}, Compute Type: {compute_type}")

    try:
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        audio = whisperx.load_audio(str(vocals_path))
        result = model.transcribe(audio, batch_size=(4 if device=="cuda" else 1), language="vi")
        
        # Save raw result
        with open(segment_out_dir / "raw_whisper.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        # Align
        print("   -> Aligning...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        del model, model_a
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ WhisperX Error: {e}")
        return

    # 3. Process & Label Segments
    print("\n[Step 3] Labeling Segments (Code-Switching)...")
    final_results = []
    ambiguous_set = {"do", "to", "on", "a", "an"} 
    
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        raw_text = segment.get("text", "").strip()
        
        # Filter noise
        if not re.search(r'[a-zA-Z0-9à-ỹÀ-Ỹ]', raw_text): continue

        if "words" not in segment:
            final_content = clean_text_content(raw_text)
            if final_content:
                final_results.append({"start": start, "end": end, "text": final_content})
            continue

        seg_words = []
        for word_obj in segment["words"]:
            if "word" not in word_obj: continue
            raw_w = word_obj["word"]
            clean_w = clean_text_content(raw_w)
            if not clean_w: continue
            
            is_vn = is_vietnamese_word(raw_w)
            lower_w = clean_w.lower()
            is_ambiguous = (lower_w in ambiguous_set) or (lower_w.isdigit() and len(lower_w) == 1)
            
            seg_words.append({"text": clean_w, "is_vn": is_vn, "is_ambiguous": is_ambiguous})

        if not seg_words: continue

        # Resolve Ambiguous
        for k, w_item in enumerate(seg_words):
            if w_item["is_ambiguous"]:
                prev_is_vn = seg_words[k-1]["is_vn"] if k > 0 else None
                next_is_vn = seg_words[k+1]["is_vn"] if k < len(seg_words) - 1 else None
                
                if (prev_is_vn is False) or (next_is_vn is False):
                    w_item["is_vn"] = False
                elif prev_is_vn is not None:
                    w_item["is_vn"] = prev_is_vn
                elif next_is_vn is not None:
                    w_item["is_vn"] = next_is_vn
                else:
                    w_item["is_vn"] = False

        # Format Text
        parts = []
        current_lbl = None
        buf = []
        for w in seg_words:
            lbl = "[vi]" if w["is_vn"] else "[en]"
            if lbl != current_lbl:
                if buf: parts.append(f" {current_lbl}{' '.join(buf)}")
                current_lbl = lbl
                buf = [w["text"]]
            else:
                buf.append(w["text"])
        if buf: parts.append(f" {current_lbl}{' '.join(buf)}")
        
        final_text = "".join(parts).lstrip()
        final_results.append({"start": start, "end": end, "text": final_text})

    # Save Full Transcript
    full_transcript_path = segment_out_dir / "full_transcript.txt"
    with open(full_transcript_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(final_results):
            f.write(f"[SEG-{i+1:03d}] [{item['start']:.2f}-{item['end']:.2f}]: {item['text']}\n")
    print(f"✅ Full transcript saved to: {full_transcript_path}")

    # 4. Cut Segments (With Fixes)
    print(f"\n[Step 4] Cutting {len(final_results)} segments based on aligned timestamps...")
    
    try:
        waveform, sr = torchaudio.load(str(vocals_path))
    except Exception as e:
        print(f"❌ Critical Error: Could not load vocals file {vocals_path} for cutting. {e}")
        return

    duration_sec = waveform.shape[1] / sr
    pad_sec = 0.05 
    
    success_count = 0
    fail_count = 0

    for i, item in enumerate(final_results):
        start = item["start"]
        end = item["end"]
        
        # --- FIX: Validate Timestamps ---
        if start >= end:
            print(f"⚠️ Warning: Segment {i+1} has invalid duration (Start: {start}, End: {end}). Skipping.")
            fail_count += 1
            continue
            
        start_cut = max(0.0, start - pad_sec)
        end_cut = min(duration_sec, end + pad_sec)
        
        if start_cut >= end_cut:
             print(f"⚠️ Warning: Segment {i+1} calculated cut is invalid. Skipping.")
             fail_count += 1
             continue

        start_frame = int(start_cut * sr)
        end_frame = int(end_cut * sr)
        
        # Boundary check
        if end_frame > waveform.shape[1]: end_frame = waveform.shape[1]

        cut_wav = waveform[:, start_frame:end_frame]
        
        # --- FIX: Check Empty Audio ---
        if cut_wav.shape[1] == 0:
            print(f"⚠️ Warning: Segment {i+1} resulted in empty audio. Skipping.")
            fail_count += 1
            continue
            
        seg_name = f"segment_{i+1:03d}.wav"
        seg_path = segment_out_dir / seg_name
        
        try:
            # --- FIX: Explicit Format ---
            torchaudio.save(str(seg_path), cut_wav, sr, format="wav")
            
            # Save individual text
            txt_path = segment_out_dir / f"segment_{i+1:03d}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"[SEG-{i+1:03d}] [{start:.2f}-{end:.2f}]: {item['text']}")
            
            success_count += 1
        except Exception as e:
            print(f"❌ Error saving segment {i+1}: {e}")
            fail_count += 1
            
    print(f"\n✅ Completed! Success: {success_count}, Failed: {fail_count}")
    print(f"   -> Output: {segment_out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Run Transcription & Segmentation.")
    parser.add_argument("input_name", nargs="?", help="Name of the input file (excluding extension) or path to original file.")
    parser.add_argument("--output_dir", default="ASRmodel", help="Root output directory")
    
    args = parser.parse_args()
    
    output_dir_path = Path(args.output_dir)
    target_name = args.input_name
    
    # Auto-detect if not provided
    if not target_name:
        print(f"🔹 No input name specified. Checking '{output_dir_path}/temp/htdemucs' for potential inputs...")
        htdemucs_dir = output_dir_path / "temp" / "htdemucs"
        
        candidates = []
        if htdemucs_dir.exists():
            for d in htdemucs_dir.iterdir():
                if d.is_dir() and (d / "vocals.wav").exists():
                    candidates.append(d.name)
        
        if not candidates:
            print("❌ No pre-processed vocals found. Run 'run_demucs.py' first.")
            return
            
        # Pick the most recently modified one or just the first
        target_name = candidates[0] 
        print(f"   -> Auto-selected based on existing folder: {target_name}")

    # Handle different input path scenarios
    target_path = Path(target_name)
    
    if target_path.exists() and target_path.is_file():
        # Scenario A: User passes "ASRmodel/.../IELTS/vocals.wav"
        if target_path.name == "vocals.wav":
            # Use the parent folder name (e.g., "IELTS")
            target_name = target_path.parent.name
        else:
            # Scenario B: User passes "IELTS.mp3" -> Use stem "IELTS"
            target_name = target_path.stem
            
    elif len(target_path.parts) > 1:
         # Scenario C: User passes path that might not exist or relative path "some/dir/IELTS"
         target_name = target_path.name

    print(f"🔹 Processing Project Name: {target_name}")
    run_transcription(target_name, output_dir_path)

if __name__ == "__main__":
    main()
