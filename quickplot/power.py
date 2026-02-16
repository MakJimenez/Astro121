# File: power.py
import numpy as np
import matplotlib.pyplot as plt

def pwr(data, index, sample_rate, xmin=0, xmax=1421):
	"""
	Pwr takes in args (data, index, sample_rate, xmin, and xmax)
	data : the file name as a string with .npz at the end
	index: integer number within block range of data taken. avoid 0.
	sample_rate : sample freqeucny, in MHz
	"""

	readdata = np.load(data)
	notes = readdata["arr_0"]
	print(notes)
	indexdata = readdata["arr_1"]

	N = len(indexdata[index])

	plt.figure(figsize=(10,4))
	shifted_ft = np.fft.fftshift(np.fft.fft(bird7[3]))
	power = np.abs(shifted_ft) ** 2
	f = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))

	plt.plot(f / 1e3, power) # 1e3 is unit conversion from MHz to kHz
	plt.xlim(xmin,xmax)
	plt.xlabel("Frequency")
	plt.ylabel("Power")
	plt.title("Power Spectrum")
	return(plt.show())
