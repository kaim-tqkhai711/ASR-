import os
import sys
import subprocess
import shutil
import warnings
import json
import re
from pathlib import Path
import numpy as np
import torch
import torchaudio

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
INPUT_FILE = Path("input1.wav") 
OUTPUT_DIR = Path("ASRmodel")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_python_script_path(cmd_name):
    scripts_dir = Path(sys.prefix) / "Scripts"
    exe_path = scripts_dir / f"{cmd_name}.exe"
    if exe_path.exists(): return str(exe_path)
    script_path = scripts_dir / cmd_name
    if script_path.exists(): return str(script_path)
    path_from_shutil = shutil.which(cmd_name)
    if path_from_shutil: return path_from_shutil
    return cmd_name

def clean_text_content(text):
    """
    Remove punctuation and symbols, normalize spaces.
    Keep basic alphanumeric and Vietnamese chars.
    """
    text = re.sub(r'[^\w\s]', '', text) 
    return text.strip()

def is_vietnamese_word(word_text):
    """
    Heuristic cải tiến để xác định từ tiếng Việt.
    Trả về: True (VI), False (EN).
    (Logic Context sẽ được xử lý ở vòng lặp chính cho các từ Ambiguous)
    """
    chars_to_strip = '.,?!:;«»""\'\'()[]{}-'
    clean_word = word_text.strip(chars_to_strip)
    if not clean_word: return False
    w_lower = clean_word.lower()
    
    # 1. Check dấu (mạnh nhất)
    if not clean_word.isascii(): return True
    
    # 2. Check cấu trúc Tiếng Anh rõ ràng
    invalid_endings = ('f', 'j', 'l', 'r', 's', 'v', 'z', 'w', 'k', 'd', 'b') 
    if w_lower.endswith(invalid_endings): return False
        
    # Từ tiếng Anh phổ biến (Blacklist)
    # REMOVED "do", "to", "on", "so" as requested
    en_blacklist = {
        "ok", "okay", "video", "clip", "view", "like", "comment", "share", 
        "subscribe", "channel", "fan", "anti", "game", "show", "live",
        "is", "am", "are", "does", "did", "have", "has", "had", 
        "but", "and", "or", "for", "in", "at", "of", "by", "my", "me", "he", "she", "it", "we", "they",
        "go", "see", "look", "watch", "eat", "drink", "rice", "noodle", "food"
    }
    if w_lower in en_blacklist: return False

    # 3. Whitelist mở rộng
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
# MAIN PIPELINE
# ==========================================

def run_pipeline():
    print(f"🔹 File đầu vào: {INPUT_FILE}")
    print(f"🔹 Thư mục output gốc: {OUTPUT_DIR.resolve()}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    segment_out_dir = OUTPUT_DIR / "output_segment"
    segment_out_dir.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print(f"❌ Lỗi: Không tìm thấy file {INPUT_FILE}")
        return

    # STEP 1: DEMUCS
    print("\n[Step 1] Tách Vocal bằng Demucs...")
    input_stem = INPUT_FILE.stem
    vocals_path = temp_dir / "htdemucs" / input_stem / "vocals.wav"
    
    if vocals_path.exists():
        print("   -> File vocals đã tồn tại, sẽ sử dụng lại.")
    else:
        demucs_exe = get_python_script_path("demucs")
        demucs_cmd = [demucs_exe, "-n", "htdemucs", "--two-stems=vocals", str(INPUT_FILE), "-o", str(temp_dir)]
        try:
            print(f"   -> Running Demucs: {demucs_cmd}")
            subprocess.run(demucs_cmd, check=True)
        except Exception as e:
            print(f"❌ Lỗi Demucs: {e}")
            return
    if not vocals_path.exists(): return

    # STEP 2: SKIP DEEPFILTERNET
    print("\n[Step 2] Bỏ qua Speech Enhancement (DeepFilterNet).")
    cleaned_audio_path = vocals_path

    # STEP 3: WHISPERX
    print("\n[Step 3] Nhận dạng toàn bộ file (WhisperX)...")
    import whisperx
    import gc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        audio = whisperx.load_audio(str(cleaned_audio_path))
        result = model.transcribe(audio, batch_size=(4 if device=="cuda" else 1), language="vi")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        del model, model_a
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Lỗi WhisperX: {e}")
        return

    # STEP 4: LABELING WITH CONTEXT AWARENESS
    print("\n[Step 4] Gán nhãn Code-Switching (Context-Aware)...")
    final_results = []
    
    ambiguous_set = {"do", "to", "on"} 
    
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        raw_text = segment.get("text", "").strip()
        
        if not re.search(r'[a-zA-Z0-9à-ỹÀ-Ỹ]', raw_text):
            continue

        if "words" not in segment:
            final_content = clean_text_content(raw_text)
            if not final_content: continue
            final_results.append({"start": start, "end": end, "text": final_content})
            continue

        # 1. Collect all valid words
        seg_words = []
        for word_obj in segment["words"]:
            if "word" not in word_obj: continue
            raw_w = word_obj["word"]
            clean_w = clean_text_content(raw_w)
            if not clean_w: continue
            
            # Determine initial status
            is_vn = is_vietnamese_word(raw_w)
            is_ambiguous = (clean_w.lower() in ambiguous_set)
            
            seg_words.append({
                "text": clean_w,
                "is_vn": is_vn,
                "is_ambiguous": is_ambiguous
            })
            
        if not seg_words: continue

        # 2. Resolve Ambiguous Words based on Bidirectional Context
        for k, w_item in enumerate(seg_words):
            if w_item["is_ambiguous"]:
                
                # Context Vectors
                prev_vi = None
                if k > 0:
                    prev_vi = seg_words[k-1]["is_vn"]
                
                next_vi = None
                if k < len(seg_words) - 1:
                    next_vi = seg_words[k+1]["is_vn"]
                
                # Logic Quyết định: 
                # Ưu tiên 1: Lấy theo từ phía trước (Forward chaining)
                if prev_vi is not None:
                     w_item["is_vn"] = prev_vi
                     
                # Ưu tiên 2: Nếu không có từ trước (đầu câu), lấy theo từ phía sau (Backward chaining)
                elif next_vi is not None:
                     w_item["is_vn"] = next_vi
                
                # Ưu tiên 3: Nếu cả 2 đều không có (câu 1 từ), hoặc context xung quanh cũng ambiguous -> Default False (En)
                else:
                    w_item["is_vn"] = False
        
        # 3. Build Formatted String
        formatted_line_parts = []
        current_label = None
        buffer_text = []

        for w_item in seg_words:
            label = "[vi]" if w_item["is_vn"] else "[en]"
            text = w_item["text"]
            
            if label != current_label:
                if buffer_text:
                    formatted_line_parts.append(f" {current_label}{' '.join(buffer_text)}")
                current_label = label
                buffer_text = [text]
            else:
                buffer_text.append(text)
        
        if buffer_text:
            formatted_line_parts.append(f" {current_label}{' '.join(buffer_text)}")
            
        final_content = "".join(formatted_line_parts).lstrip()
        
        final_results.append({
            "start": start,
            "end": end,
            "text": final_content,
        })
    
    # OUTPUT AND CUTTING (unchanged logic)
    # Saving Full Transcript
    full_transcript_path = segment_out_dir / "full_transcript.txt"
    with open(full_transcript_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(final_results):
             f.write(f"[SEG-{i+1:03d}] [{item['start']:.2f}-{item['end']:.2f}]: {item['text']}\n")
    print(f"✅ Đã lưu transcript tổng tại: {full_transcript_path}")

    print(f"\n[Step 5] Cắt {len(final_results)} segment wav...")
    waveform, sr = torchaudio.load(str(cleaned_audio_path))
    duration_sec = waveform.shape[1] / sr
    pad_sec = 0.05 
    
    for i, item in enumerate(final_results):
        start = item["start"]
        end = item["end"]
        # Padding
        start_cut = max(0.0, start - pad_sec)
        end_cut = min(duration_sec, end + pad_sec)
        
        start_frame = int(start_cut * sr)
        end_frame = int(end_cut * sr)
        
        if end_frame > waveform.shape[1]: end_frame = waveform.shape[1]
        cut_wav = waveform[:, start_frame:end_frame]
        
        seg_name = f"segment_{i+1:03d}.wav"
        seg_path = segment_out_dir / seg_name
        torchaudio.save(str(seg_path), cut_wav, sr)
        
        txt_path = segment_out_dir / f"segment_{i+1:03d}.txt"
        full_line = f"[SEG-{i+1:03d}] [{start:.2f}-{end:.2f}]: {item['text']}"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_line)

    print(f"✅ Hoàn tất! Output tại: {segment_out_dir.resolve()}")

if __name__ == "__main__":
    run_pipeline()
