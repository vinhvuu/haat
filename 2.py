import tkinter as tk
from threading import Thread
import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import sys

# ================== CONFIG ==================
SAMPLE_RATE = 16000
BLOCK_SIZE = 3   # block 3 giây
DEVICE = "cpu"  # hoặc "cuda" nếu có GPU
MODEL_NAME = "small"  # Dùng Whisper small (public)
# ============================================

print("🔄 Đang load Whisper model...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")
print("✅ Model loaded")

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def transcribe_loop(text_widget):
    audio_buffer = np.array([], dtype=np.float32)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while True:
            data = q.get()
            audio_buffer = np.concatenate((audio_buffer, data[:, 0]))

            if len(audio_buffer) >= SAMPLE_RATE * BLOCK_SIZE:
                block = audio_buffer[:SAMPLE_RATE * BLOCK_SIZE]
                audio_buffer = audio_buffer[SAMPLE_RATE * BLOCK_SIZE:]

                volume = np.mean(np.abs(block))
                if volume < 0.005:
                    continue

                segments, info = model.transcribe(block, beam_size=5, language="vi")
                text_out = " ".join([seg.text for seg in segments])
                if text_out.strip():
                    text_widget.insert(tk.END, text_out + "\n")
                    text_widget.see(tk.END)

def main():
    root = tk.Tk()
    root.title("Realtime Whisper Transcription")

    text_box = tk.Text(root, wrap="word", font=("Arial", 14))
    text_box.pack(expand=True, fill="both")

    thread = Thread(target=transcribe_loop, args=(text_box,), daemon=True)
    thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()
