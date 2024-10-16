import sys
import rnnoise
import sounddevice as sd
import queue
import io
from faster_whisper import WhisperModel
import wave

VAD = 0.9
SAMPLERATE = 48000
denoiser = rnnoise.RNNoise()
denoise_q = queue.Queue()
denoised_data = b''
model = WhisperModel("base", device="cpu", compute_type="int8")
tmp_wav = "/tmp/asr.wav"
finish_cycle = 20
counter = 0

def stt(data):
    with wave.open(tmp_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLERATE)
        wf.writeframes(data)

    segments, info = model.transcribe(tmp_wav, beam_size=5, language="tr")
    result = ""
    for segment in segments:
        result += " " + segment.text
    return result

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    denoise_q.put(indata.copy())

with sd.InputStream(samplerate=SAMPLERATE, channels=1, blocksize = 480 ,dtype='int16', callback=audio_callback):
    while True:
        data = denoise_q.get()
        vad ,denoised = denoiser.process_frame(data)
        print("\r" + "{:.2f}".format(vad), end="")
        if vad > VAD or counter < finish_cycle:
            denoised_data += bytes(denoised)
        else:
            if len(denoised_data) > SAMPLERATE:
                print(stt(denoised_data))
            denoised_data =  b''
        
        if vad > VAD:
            counter = 0
        else:
            counter += 1

denoiser.destroy()
