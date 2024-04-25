import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
import psutil
import time

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

cpu_usage_history = []

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
    
    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=reduced_data, sr=RATE, n_mels=128, fmax=8000)
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    # Measure CPU usage
    cpu_percent = psutil.cpu_percent()
    
    # Add CPU usage to history
    cpu_usage_history.append(cpu_percent)
    
    # Clear the current spectrogram
    ax.clear()
    
    # Display the new mel spectrogram
    librosa.display.specshow(S_DB, sr=RATE, x_axis='time', y_axis='mel', ax=ax)
    plt.title('Live Mel Spectrogram with Noise Reduction')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')

ani = animation.FuncAnimation(fig, update, init_func=init, blit=False, interval=50)

try:
    plt.show()
except KeyboardInterrupt:
    pass

# Stop and close audio stream
stream.stop_stream()
stream.close()
audio.terminate()

# Calculate maximum change in CPU usage
max_cpu_change = max(cpu_usage_history) - min(cpu_usage_history)
print("Maximum change in CPU usage:", max_cpu_change, "%")
 