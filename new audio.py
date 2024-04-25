import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa

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

def init():
    # This function will run once at the beginning of the animation
    pass

def update(frame):
    # Read audio stream
    data = stream.read(CHUNK, exception_on_overflow=False)
    # Convert to NumPy array
    np_data = np.frombuffer(data, dtype=np.float32)
    
    # Apply noise reduction using librosa
    reduced_data = librosa.effects.preemphasis(np_data)
    
    # Calculate the STFT
    D = librosa.amplitude_to_db(np.abs(librosa.stft(reduced_data)), ref=np.max)
    
    # Clear the current spectrogram
    ax.clear()
    
    # Display the new spectrogram
    librosa.display.specshow(D, sr=RATE, x_axis='time', y_axis='log', ax=ax)
    plt.title('Live Spectrogram with Noise Reduction')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

ani = animation.FuncAnimation(fig, update, init_func=init, blit=False, interval=50)

plt.show()

stream.stop_stream()
stream.close()
audio.terminate()