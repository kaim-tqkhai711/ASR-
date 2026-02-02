import whisperx
import sys

print(f"WhisperX file: {whisperx.__file__}")
print(f"WhisperX dir: {dir(whisperx)}")

try:
    import whisperx.vad
    print("SUCCESS: import whisperx.vad works")
except ImportError as e:
    print(f"FAILURE: import whisperx.vad failed: {e}")

try:
    from whisperx import vad
    print("SUCCESS: from whisperx import vad works")
except ImportError as e:
    print(f"FAILURE: from whisperx import vad failed: {e}")
