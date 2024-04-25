'''
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
from matplotlib.table import Table

# PyAudio parameters
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 44100 * 20  # 20-second chunk

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

# Create an empty table with 14 rows (13 MFCC coefficients + header row) and 2 columns (Coefficient, Value)
data = [['MFCC Coefficient', 'Value']] + [['MFCC' + str(i), ''] for i in range(13)]
num_rows = len(data)
num_cols = len(data[0])

# Create figure and axes for the table
fig, ax = plt.subplots()
ax.axis('off')  # Turn off axis for the table

# Create an empty table
table = Table(ax, cellText=data, loc='center', cellLoc='center', colWidths=[0.2, 0.2])

# Set table properties
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)  # Scale the table for better readability

# Adjust table layout
table_props = table.properties()
table_props.update({'cellLoc': 'center', 'cellColours': [['lightgray'] * num_cols] * num_rows})

def update_table(frame):
    # Read audio stream
    data = stream.read(CHUNK, exception_on_overflow=False)
    # Convert to NumPy array
    np_data = np.frombuffer(data, dtype=np.float32)

    # Apply noise reduction using librosa
    reduced_data = librosa.effects.preemphasis(np_data)

    # Calculate the MFCC coefficients for the 20-second chunk
    mfccs = librosa.feature.mfcc(y=reduced_data, sr=RATE, n_mfcc=13)

    # Update the table with the latest MFCC coefficients
    for i in range(13):
        table[(i + 1, 1)].get_text().set_text(f'{mfccs[i, 0]:.2f}')  # Displaying the first MFCC coefficient

    return [table]

ani = animation.FuncAnimation(fig, update_table, blit=True, interval=1000)  # Update every second for a 20-second chunk

plt.show()

stream.stop_stream()
stream.close()
audio.terminate()

'''
