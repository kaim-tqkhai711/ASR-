# Code-Switching ASR Pipeline

Pipeline này thực hiện xử lý âm thanh đầu vào, tách lời (vocals), nhận dạng giọng nói (ASR) sử dụng WhisperX và gán nhãn code-switching (Tiếng Việt/Tiếng Anh) cho kết quả.

## Yêu cầu hệ thống

1.  **Python 3.8+**
2.  **FFmpeg**: Cần cài đặt và thêm vào PATH của hệ thống.
3.  **CUDA (Optional)**: Khuyến nghị sử dụng GPU NVIDIA để tốc độ xử lý nhanh hơn (WhisperX chạy tốt nhất trên GPU).

## Cài đặt

Cài đặt các thư viện Python cần thiết:

```bash
# Cài đặt PyTorch (Chọn phiên bản phù hợp với hệ thống của bạn tại https://pytorch.org/)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt Demucs
pip install demucs

# Cài đặt WhisperX
pip install git+https://github.com/m-bain/whisperx.git

# Cài đặt các thư viện khác
pip install numpy
```

## Chuẩn bị dữ liệu

1.  Chuẩn bị file âm thanh đầu vào (định dạng `.wav`).
2.  Đặt tên file là `input1.wav` và để ở thư mục gốc của project (cùng cấp với file `run_pipeline_v2.py`).
    *   *Lưu ý*: Nếu muốn thay đổi tên file hoặc đường dẫn, hãy mở file `run_pipeline_v2.py` và chỉnh sửa biến `INPUT_FILE` ở phần CONFIGURATION (dòng 25).

## Cách chạy

Mở terminal tại thư mục dự án và chạy lệnh:

```bash
python run_pipeline_v2.py
```

## Luồng xử lý

Script sẽ thực hiện các bước sau:
1.  **Tách Vocal**: Sử dụng `demucs` để tách giọng hát/nói khỏi nhạc nền (nếu có). File vocal sẽ được lưu tạm tại `ASRmodel/temp`.
2.  **ASR (WhisperX)**: Sử dụng mô hình `large-v2` để chuyển đổi âm thanh thành văn bản và căn chỉnh thời gian (alignment).
3.  **Gán nhãn (Labeling)**: Áp dụng thuật toán heuristic để phân loại từ là Tiếng Việt `[vi]` hoặc Tiếng Anh `[en]`.
4.  **Xuất kết quả**: Cắt file âm thanh thành các segment nhỏ và lưu transcript.

## Output

Kết quả sẽ được lưu trong thư mục `ASRmodel/output_segment`:
*   `full_transcript.txt`: File chứa toàn bộ nội dung transcript với nhãn ngôn ngữ và timestamp.
*   `segment_XXX.wav`: Các đoạn audio clip đã được cắt nhỏ theo câu.
*   `segment_XXX.txt`: Transcript tương ứng cho từng đoạn clip.
