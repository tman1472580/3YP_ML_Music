import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyaudio

# Constants
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Initialize plot
fig, ax = plt.subplots()
x = np.arange(0, 2 * CHUNK, 2)
line, = ax.plot(x, np.random.rand(CHUNK))

def update_plot(frame):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    line.set_ydata(data)
    return line,

ani = FuncAnimation(fig, update_plot, blit=False)
plt.show()