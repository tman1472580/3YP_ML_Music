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

# Create figure and axes for the plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

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
    
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=13)
    
    # Clear the current plots
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    # Display the new mel spectrogram
    librosa.display.specshow(S_DB, sr=RATE, x_axis='time', y_axis='mel', ax=ax1)
    ax1.set_title('Live Mel Spectrogram with Noise Reduction')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mel Frequency')
    
    # Display the MFCCs
    librosa.display.specshow(mfccs, sr=RATE, x_axis='time', ax=ax2)
    ax2.set_title('MFCCs')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('MFCC')
    
    # Display the waveform
    ax3.plot(reduced_data)
    ax3.set_title('Waveform')
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Amplitude')

ani = animation.FuncAnimation(fig, update, init_func=init, blit=False, interval=50)

plt.show()

stream.stop_stream()
stream.close()
audio.terminate()
