import pprint
import IPython.display as ipd
import torch
import librosa
import timeit
import numpy as np



mars5, config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)

start = timeit.default_timer()
wav, sr = librosa.load('./example.wav', 
                       sr=mars5.sr, mono=True)
wav = torch.from_numpy(wav)
ref_transcript = "We actually haven't managed to meet demand."
# wav, sr = librosa.load('./Mahesh_trim.wav', 
#                        sr=mars5.sr, mono=True)
# wav = torch.from_numpy(wav)
# ref_transcript = "AI, till now has been an intellectual language, from now it will be a common language."
print("Reference audio:")
ipd.display(ipd.Audio(wav.numpy(), rate=mars5.sr))
print(f"Reference transcript: {ref_transcript}")

deep_clone = True # set to False if you don't know prompt transcript or want fast inference.
# Below you can tune other inference settings, like top_k, temperature, top_p, etc...
cfg = config_class(deep_clone=deep_clone, rep_penalty_window=100,
                      top_k=100, temperature=0.7, freq_penalty=3)

ar_codes, wav_out = mars5.tts("Hi, my name is Saikrishna Gutha. I am from Andhra.", wav, 
          ref_transcript,
          cfg=cfg)

print('Synthesized output audio:')
ipd.Audio(wav_out.numpy(), rate=mars5.sr)

np.save("wav_out.npy",wav_out.numpy())
ar_wav = mars5.vocode(ar_codes.cpu()[:, None])
ipd.Audio(ar_wav.numpy(), rate=mars5.sr)

np.save("ar_wav.npy",ar_wav.numpy())

stop = timeit.default_timer()


print('Time: ', stop - start) 


from scipy.io.wavfile import write

# Assuming wav_out and ar_wav are NumPy arrays with audio data
# Convert to NumPy array if needed
wav_out_np = wav_out.numpy()  # for the first audio output
ar_wav_np = ar_wav.numpy()  # for the second audio output


# Save the audio files
write('synthesized_output.wav', mars5.sr, wav_out_np.astype(np.float32))
write('vocode_output.wav', mars5.sr, ar_wav_np.astype(np.float32))

print('Audio files saved successfully.')

