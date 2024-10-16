
import numpy
import torch
import queue
import sys
import sounddevice

SAMPLE_RATE = 16000

torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

denoise_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    denoise_q.put(indata.copy())

with sounddevice.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize = 512 ,dtype='float32', callback=audio_callback):
    while True:
        raw = denoise_q.get()
        audio_data = numpy.mean(raw, axis=1)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        confidence = model(audio_tensor, SAMPLE_RATE).item()
        print(confidence)
