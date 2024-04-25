import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
import librosa.display
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime


FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024


audio = pyaudio.PyAudio()


engine = create_engine('sqlite:///audio_data.db')
Base = declarative_base()

class AudioData(Base):
    __tablename__ = 'audio_data'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    data = Column(String)  

Base.metadata.create_all(engine)

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)


Session = sessionmaker(bind=engine)
session = Session()

fig, ax = plt.subplots()

def init():
    
    pass

def update(frame):
    
    data = stream.read(CHUNK, exception_on_overflow=False)
    
    np_data = np.frombuffer(data, dtype=np.float32)

   
    reduced_data = librosa.effects.preemphasis(np_data)

   
    D = librosa.amplitude_to_db(np.abs(librosa.stft(reduced_data)), ref=np.max)

    
    audio_entry = AudioData(data=str(np_data.tolist()))
    session.add(audio_entry)
    session.commit()

    
    ax.clear()

    
    librosa.display.specshow(D, sr=RATE, x_axis='time', y_axis='log', ax=ax)
    plt.title('Live Spectrogram with Noise Reduction')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

ani = animation.FuncAnimation(fig, update, init_func=init, blit=False, interval=50)

plt.show()

stream.stop_stream()
stream.close()
audio.terminate()
