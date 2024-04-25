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
CHUNK = 1024  # Number of samples per frame

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
    plt.title('Live Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    global im
    # Initialize a placeholder for the spectrogram
    im = ax.imshow(np.zeros((int(CHUNK/2), CHUNK)), aspect='auto', origin='lower',
                   extent=[0, CHUNK/RATE, 0, RATE/2], cmap='viridis')
    fig.colorbar(im, ax=ax, format='%+2.0f dB')

def update(frame):
    # Read audio stream
    data = stream.read(CHUNK, exception_on_overflow=False)
    # Convert to NumPy array
    np_data = np.frombuffer(data, dtype=np.float32)
    
    # Compute the spectrogram
    D = librosa.stft(np_data, n_fft=2048, hop_length=512)  # STFT of the signal
    S_DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert amplitude to decibels
    
    # Update the spectrogram display
    im.set_data(S_DB)
    im.set_clim(vmin=S_DB.min(), vmax=S_DB.max())

    return im,

ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

plt.show()

stream.stop_stream()
stream.close()
audio.terminate()
