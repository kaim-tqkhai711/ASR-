# Hướng dẫn chạy ASR Pipeline trên Google Colab

Đây là hướng dẫn chi tiết để chạy pipeline nhận dạng giọng nói (ASR) code-switching trên Google Colab.

## Bước 1: Chuẩn bị Môi trường

1.  Truy cập [Google Colab](https://colab.research.google.com/).
2.  Tạo một Notebook mới.
3.  **Quan trọng**: Bật GPU.
    *   Vào menu **Runtime** (Thời gian chạy) > **Change runtime type** (Thay đổi loại thời gian chạy).
    *   Chọn **T4 GPU** (hoặc GPU khác nếu có sẵn).
    *   Nhấn **Save**.

## Bước 2: Cài đặt Thư viện

Copy và chạy đoạn code sau trong một cell để cài đặt các thư viện cần thiết:

```python
# Cài đặt FFmpeg
!apt-get install -y ffmpeg

# Cài đặt các thư viện Python (Phiên bản cụ thể để tránh lỗi tương thích)
# Numpy < 2.0 quan trọng cho WhisperX
!pip install "numpy<2.0" 

# PyTorch 2.1.2 + CUDA 11.8 (Ổn định trên Colab T4)
!pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Các thư viện xử lý âm thanh
!pip install demucs==4.0.1 

# WhisperX (Cài từ source nhưng sẽ dùng numpy/torch đã cài ở trên)
!pip install git+https://github.com/m-bain/whisperx.git

print("✅ Đã cài đặt xong các thư viện!")
```

## Bước 3: Upload File

Bạn cần upload file script `run_pipeline_v2.py` và file âm thanh của bạn (ví dụ `2005.wav`).

1.  Nhấn vào biểu tượng **Thư mục (Files)** ở thanh bên trái.
2.  Kéo thả file `run_pipeline_v2.py` và file âm thanh vào đó.
3.  *Lưu ý: File upload sẽ bị mất khi tắt Colab, nên hãy lưu lại kết quả sau khi chạy xong.*

## Bước 4: Chạy Pipeline

Mặc định script đang set `INPUT_FILE = Path("2005.wav")`.
Nếu file của bạn có tên khác, hãy chạy lệnh dưới đây để tự động sửa tên file trong script và chạy pipeline:

```python
# @title Cấu hình và Chạy
import os

# Đổi tên file audio của bạn ở đây
AUDIO_FILENAME = "2005.wav" # <-- Sửa tên này nếu file của bạn khác

# Kiểm tra file tồn tại
if not os.path.exists(AUDIO_FILENAME):
    print(f"❌ Không tìm thấy file {AUDIO_FILENAME}. Hãy upload file vào mục Files bên trái.")
else:
    # Sửa tên file trong script run_pipeline_v2.py dùng lệnh sed
    # Lệnh này sẽ thay thế dòng INPUT_FILE = Path("...") thành file bạn chọn
    !sed -i "s/INPUT_FILE = Path(\".*\")/INPUT_FILE = Path(\"{AUDIO_FILENAME}\")/" run_pipeline_v2.py
    
    print(f"🚀 Đang chạy pipeline cho file: {AUDIO_FILENAME}...")
    !python run_pipeline_v2.py
```

## Bước 5: Tải xuống Kết quả

Sau khi chạy xong, kết quả sẽ nằm trong thư mục `ASRmodel`. Bạn có thể nén lại để tải về cho dễ.

```python
import shutil
from google.colab import files

output_folder = "ASRmodel"

if os.path.exists(output_folder):
    # Nén thư mục thành zip
    shutil.make_archive("ket_qua_asr", 'zip', output_folder)
    print("✅ Đã nén xong. Đang tải xuống...")
    
    # Tự động tải xuống
    files.download("ket_qua_asr.zip")
else:
    print("⚠️ Không tìm thấy folder output. Có lỗi khi chạy script?")
```
