import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pydub import AudioSegment

# file_path = 'sounds/A440.wav'
file_path = 'sounds/timer.wav'
# file_path = 'sounds/sweep.wav'
# file_path = 'sounds/believer.wav'

# import a mono wavfile as the source signal
# the sampling frequency should match that of the room
from scipy.io import wavfile

# Function to convert a stereo file to a mono file
def stereo_to_mono_array(file_path):
    # Load stereo audio file
    audio = AudioSegment.from_wav(file_path)

    # Convert stereo to mono
    mono_audio = audio.set_channels(1)

    # Export mono audio to WAV file
    mono_audio.export(f"{file_path}_mono.wav", format="wav")

stereo_to_mono_array(file_path)
fs, audio = wavfile.read(f"{file_path}_mono.wav")
# The desired reverberation time and dimensions of the room
rt60_tgt = 0.5   # seconds
room_dim = [6, 6, 6]  # meters
L = 1.28     # Distance between microphones (meters)
mic_h = 1    # Assume microphone is 1m from the ground

# We invert Sabine's formula to obtain the parameters for the ISM simulator
e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

# Create the room
room = pra.ShoeBox(
    room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order,
    use_rand_ism = True, max_rand_disp = 0.05
)

# Offsets from origin: Set a 1m offset from the origin
x_offset = 1
y_offset = 1

# Source position
source_pos = [x_offset + L/4, y_offset + np.sqrt(3)*L/4, mic_h]

# place the source in the room
room.add_source(source_pos, signal=audio)
# Microphone positions
mic_1_pos  = [x_offset + 0, y_offset + 0, mic_h]
mic_2_pos  = [x_offset + L, y_offset + 0, mic_h]
mic_3_pos  = [x_offset + L/2, y_offset + np.sqrt(3)*L/2, mic_h]
mic_gt_pos = [x_offset + L/2, y_offset + np.sqrt(3)*L/6, mic_h]
print(mic_gt_pos)

# [NORMAL MICROPHONES] Define the locations of the microphones
mic_locs = np.c_[
    mic_1_pos,  # mic 1
    mic_2_pos,  # mic 2
    mic_3_pos,  # mic 3
    mic_gt_pos  # ground truth microphone
]
print(mic_locs)

# finally place the array in the room
room.add_microphone_array(mic_locs)

# Simulate the room
room.simulate()

# Output the microphone data to wav file format
room.mic_array.to_wav(
    f"generated_mics/mic_arr.wav",
    norm=True,
    bitdepth=np.int16,
)

room.compute_rir()
plt.plot(room.rir[1][0])
plt.show()

room.plot(img_order=0)
plt.show()