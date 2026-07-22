
import pyaudio
import wave
import time


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000       
CHUNK = 1024
RECORD_SECONDS = 2    
OUTPUT_FILE = "my_digit.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print(f"🎤 Recording {RECORD_SECONDS} seconds... Speak now.")
frames = []
for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("✅ Done recording.")

stream.stop_stream()
stream.close()
p.terminate()

# Save as WAV
wf = wave.open(OUTPUT_FILE, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"Saved: {OUTPUT_FILE}")