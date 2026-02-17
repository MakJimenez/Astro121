# File: power.py
import numpy as np
import matplotlib.pyplot as plt

def pwr(data, index, sample_rate, f_center=1420.115e3, f_signal=1420.405e3, xmin=-200, xmax=200):
	"""
	Pwr takes in args (data, index, sample_rate, xmin, and xmax)
	data : the file name as a string with .npz at the end
	index: integer number within block range of data taken. avoid 0.
	sample_rate : sample freqeucny, in MHz
	f_center and f_signal should be given as the signal to the third power, aka kHz.
	"""

	# Calculating location of expected peak
	f_expected = f_signal - f_center

	# Loading in data
	readdata = np.load(data)
	notes = readdata["arr_0"]
	print(notes)
	indexdata = readdata["arr_1"]
	data = indexdata[index]
	N = len(indexdata[index])

	# Plotting
	plt.figure(figsize=(10,4))
	shifted_ft = np.fft.fftshift(np.fft.fft(indexdata[index]))
	power = np.abs(shifted_ft) ** 2
	f = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))

	plt.plot(f / 1e3, power) # 1e3 is unit conversion from MHz to kHz
	plt.axvline(f_expected, linestyle=":", color="black", label=f"Expected Relative Signal at {f_expected} kHz")
	plt.xlim(xmin,xmax)
	plt.xlabel("Frequency")
	plt.ylabel("Power")
	plt.title("Power Spectrum")
	plt.legend()
	return(plt.show())
