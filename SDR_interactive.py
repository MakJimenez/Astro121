import ugradio
import pandas as pd
from rtlsdr import RtlSdr
import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt
import time

# Title of Data Trial
trial = "testinginteractiveplot" 
sample_rate = 2.9e6
block_size = 131072 
blocks = 4 

# Data Collection Information
frequency = 1420
frequency_units = "MHz"
amplitude = -30
amplitude_units = "dBm" #readout from wave generator. stored in data as ADC
gain = 3 # dB
direct = False
unix_time = ugradio.timing.unix_time()
local_time = ugradio.timing.local_time()
location = "NCH" 
direction = "8th rung from the left, pointing north (aligned with square at bottom), above red roof"
notes = "Testing interactive plot, gain = 3."

# Note: Use this text file to collect aliased data 
sdr = ugradio.sdr.SDR(direct=False, center_freq=1420.055e6, sample_rate=2.9e6, gain=3, fir_coeffs=None)
data = sdr.capture_data(131072, nblocks=4)
print(sdr)
print(data)

# Interactive plot
sample_rate = 2.9e6

# First capture to initialize 
b0 = data[1]
x0 = b0[:, 0] + 1j * b0[:, 1]
N = len(x0)

# Hanning windowing
window = np.hanning(N)
f = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))

# Plotting the plot
plt.ion()
fig, ax = plt.subplots(figsize=(10,4))

line, = ax.plot(f / 1e6, np.zeros(N))
ax.set_xlim(-500, 500)
ax.set_ylim(40, 100)
ax.set_xlabel("Frequency Offset (kHz)")
ax.set_ylabel("Power (dB)")
ax.set_title("Live Baseband Power Spectrum")
ax.grid(True)

# Reference Line (expected tone location)
ax.axvline(155, linestyle=":", color="black")

# Live update loop
for i in range(len(data)):
	d = data[i]
	x = d[:, 0] + 1j * d[:, 1]

	fft_data = np.fft.fftshift(np.fft.fft(x * window))
	power_db = 10 * np.abs(fft_data)**2

	line.set_ydata(power_db)
	fig.canvas.draw()
	fig.canvas.flush_events()

	time.sleep(5) # control update rate
plt.ioff()

 
# Saving info as numpy array
info = np.array([f"File:{trial}",f"Frequency:{frequency} {frequency_units}", f"Amplitude:{amplitude} {amplitude_units}", f"Sampling Rate:{sample_rate}", f"Direct:{direct}", f"Unix: {unix_time}", f"Local: {local_time}", f"Location: {location}", f"Direction: {direction}", f"Notes: {notes}", f"Gain: {gain}"])
print(info)

# Saving data into .npz zip file
np.savez(f"{trial}.npz", info, data)
