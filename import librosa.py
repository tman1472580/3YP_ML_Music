import librosa
import numpy as np

WAVE_OUTPUT_FILENAME = "file.wav"  # Add this line

# Load the audio file
y, sr = librosa.load(WAVE_OUTPUT_FILENAME)

# Compute MFCC features
mfcc = librosa.feature.mfcc(y=y, sr=sr)

# Print the MFCC values
print(mfcc)
print(mfcc.shape)