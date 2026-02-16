import ugradio
import pandas as pd
from rtlsdr import RtlSdr
import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt

# Title of Data Trial
trial = "birdietest24_1420.405" 
sample_rate = 2.9e6
block_size = 131072 
blocks = 4 

# Note: Use this text file to collect aliased data 
sdr = ugradio.sdr.SDR(direct=False, center_freq=1420.155e6, sample_rate=2.9e6, fir_coeffs=None)
data = sdr.capture_data(131072, nblocks=4)
print(sdr)
print(data)

# Quick Plot to make sure everything looks good
plt.figure(figsize=(10,4))
plt.plot(data[3]) # reading a block from the middle so we don't accidentally plot the "dead block"
plt.xlabel("Sampling Index")
plt.ylabel("Amplitude in ADC")
plt.title(f"{trial}")
plt.xlim(0, 50) # zooms in so we can see wave details better
print(plt.show()) # prints wave data just collected

# Quick FT Power Plot
N = len(data[3])

plt.figure(figsize=(10,4))
shifted_ft = np.fft.fftshift(np.fft.fft(data[3]))
power = np.abs(shifted_ft) ** 2
f = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))

plt.plot(f / 1e3, power) # 1e3 is unit conversion from MHz to kHz
plt.xlim(0, 1600)
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("Power")
print(plt.show())


# Data Collection Information
frequency = 1420.405
frequency_units = "MHz"
amplitude = -80
amplitude_units = "dBm" #readout from wave generator. stored in data as ADC
direct = False
unix_time = ugradio.timing.unix_time()
local_time = ugradio.timing.local_time()
location = "NCH" 
direction = "~pointing at  zenith,empty of water"
notes = "taking everything off (direct SDR to horm connection), 1420.405 MHz, -80 dB,singal generator on, chord with label = 2 "


# Saving info as numpy array
info = np.array([f"File:{trial}",f"Frequency:{frequency} {frequency_units}", f"Amplitude:{amplitude} {amplitude_units}", f"Sampling Rate:{sample_rate}", f"Direct:{direct}", f"Unix: {unix_time}", f"Local: {local_time}", f"Location: {location}", f"Direction: {direction}", f"Notes: {notes}"])
print(info)

# Saving data into .npz zip file
np.savez(f"{trial}.npz", info, data)

