import sys
import rnnoise
import sounddevice
import queue
from vosk import Model, KaldiRecognizer, SpkModel
import json

SAMPLERATE = 48000
denoiser = rnnoise.RNNoise()
denoise_q = queue.Queue()
denoised_data = b''

words = ["merhaba", "aç", "kapat", "kapatma", "naber", "iyi misin nasılsın", "robot mutfak gel", "şarj"]
words.append("[unk]")

model = Model(lang="tr")
rec = KaldiRecognizer(model, SAMPLERATE)
rec.SetWords(True)
rec.SetGrammar(json.dumps(words, ensure_ascii=False))

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    denoise_q.put(indata.copy())

with sounddevice.InputStream(samplerate=SAMPLERATE, channels=1, blocksize = 480 ,dtype='int16', callback=audio_callback):
    while True:
        data = denoise_q.get()
        vad ,denoised = denoiser.process_frame(data)
        print("\r" + "{:.2f}".format(vad), end="")
        denoised_data += bytes(denoised)
        while len(denoised_data) >= 16000:
            data = denoised_data[-16000:]
            denoised_data = denoised_data[:-16000]
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if not "result" in result:
                    continue
                print(" " + result["text"] + "\t" + str(result["result"][0]["conf"]))

denoiser.destroy()
