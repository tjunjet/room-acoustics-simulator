import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile

# Read the audio
fs, audio = wavfile.read('../A440.wav')
# Take one channel of audio
audio = audio[:, 0]

# Create a 4 by 6 metres shoe box room
room = pra.ShoeBox([4,6], fs=fs)

# Add a source somewhere in the room
room.add_source([2.5, 4.5], signal=audio)

# Create a linear array beamformer with 4 microphones
# with angle 0 degrees and inter mic distance 10 cm
R = pra.linear_2D_array([2, 1.5], 4, 0, 0.1)

print(f'Linear Array: {type(R)}')

room.add_microphone_array(pra.Beamformer(R, room.fs))

# Now compute the delay and sum weights for the beamformer
room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])
# Simulate the room
room.simulate()
# Output the microphone data to wav file format
room.mic_array.to_wav(
    f"../generated_mics/beam_arr.wav",
    norm=True,
    bitdepth=np.int16,
)

print(f'Mic Array: {room.mic_array}')

# plot the room and resulting beamformer
room.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
# plt.show()

# Analyzing the signals from reading the wav file generated
fs, audio = wavfile.read('../generated_mics/beam_arr.wav')
print(f'Microphone recordings: {audio}')
