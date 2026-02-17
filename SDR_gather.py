import ugradio
import pandas as pd
from rtlsdr import RtlSdr
import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt

# Title of Data Trial
trial = "Zenith_test3" 
sample_rate = 2.9e6
block_size = 131072 
blocks = 10

# Data Collection Information
frequency = "sky"
frequency_units = "Hz"
amplitude = 0
amplitude_units = "dBm" #readout from wave generator. stored in data as ADC
direct = False
unix_time = ugradio.timing.unix_time()
local_time = ugradio.timing.local_time()
gain = 3
location = "NCH" 
direction = "Zenith"
notes = "Trying to find 21 cm with histogram with horn pointed at zenith.First"

# Note: Use this text file to collect aliased data 
sdr = ugradio.sdr.SDR(direct=False, center_freq=1420.395e6, sample_rate=2.9e6, gain=3, fir_coeffs=None)
data = sdr.capture_data(131072, nblocks=10)
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
f_center = 1420.395e3
f_expec = 1420e3 - f_center

plt.figure(figsize=(10,4))
shifted_ft = np.fft.fftshift(np.fft.fft(data[3]))
power = np.abs(shifted_ft) ** 2
f = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))

plt.plot(f / 1e3, power) # 1e3 is unit conversion from GHz to MHz
plt.axvline(x=f_expec, linestyle=":", color="black")
plt.xlim(0, 1600)
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("Power")
print(plt.show())

# Saving info as numpy array
info = np.array([f"File:{trial}",f"Frequency:{frequency} {frequency_units}", f"Amplitude:{amplitude} {amplitude_units}", f"Sampling Rate:{sample_rate}", f"Direct:{direct}", f"Unix: {unix_time}", f"Local: {local_time}",f"Gain: {gain}", f"Location: {location}", f"Direction: {direction}", f"Notes: {notes}"])
print(info)

# Saving data into .npz zip file
np.savez(f"{trial}.npz", info, data)
