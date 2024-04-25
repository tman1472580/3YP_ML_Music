import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
import librosa.display

# PyAudio parameters
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

fig, ax = plt.subplots()

# Placeholder for the spectrogram image
im = None

def init():
    # This function will run once at the beginning of the animation
    # It sets up the plot environment.
    plt.title('Live Mel Spectrogram with Noise Reduction')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency (Hz)')
    global im
    im = ax.imshow(np.zeros((128, CHUNK)), aspect='auto', origin='lower', 
                   extent=[0, CHUNK/RATE, 0, 8000], cmap='viridis')
    fig.colorbar(im, ax=ax, format='%+2.0f dB')

def update(frame):
    # Read audio stream
    data = stream.read(CHUNK, exception_on_overflow=False)
    # Convert to NumPy array
    np_data = np.frombuffer(data, dtype=np.float32)
    
    # Apply noise reduction using librosa
    reduced_data = librosa.effects.preemphasis(np_data)
    
    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=reduced_data, sr=RATE, n_mels=128, fmax=8000)
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    # Update the spectrogram display
    im.set_data(S_DB)
    im.set_clim(vmin=S_DB.min(), vmax=S_DB.max())

    return im,

ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

plt.show()

stream.stop_stream()
stream.close()
audio.terminate()
