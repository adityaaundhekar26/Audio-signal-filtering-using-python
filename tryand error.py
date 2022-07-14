import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import read


import winsound
from pylab import *
import scipy.signal as signal
from scipy import fftpack




#Origninal Audio signal import
samplerate, data_original = read(r"SY-IC-A 13.wav")  #import orignal signal
print(samplerate)           #sample rate=48000
duration = len(data_original) / samplerate
time = np.arange(0, duration, 1 / samplerate)  #[0.00000000e+00 2.08333333e-05 4.16666667e-05 ... 2.98660417e+00
 #                                                             2.98662500e+00 2.98664583e+00]
plt.title("Original Time Domain")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.plot(time, data_original[:,])
plt.show()
winsound.PlaySound(r'SY-IC-A 13.wav', winsound.SND_FILENAME)

transform = np.fft.fft(data_original[:,])
N = len(data_original)
n = np.arange(N)
freq = n*(samplerate/N)
mag = abs(transform)

plt.xlim(0,4000)
plt.title("FFT Of Original Signal")
plt.xlabel("Freq")
plt.ylabel("Magnitude")
plt.plot(freq, mag)
plt.show()

#Noisy Audio

samplerate, data = read(r"C:\Users\adity\Desktop\course project\noisySYA13new (1).wav") #reading .wav file
duration = len(data) / samplerate
time = np.arange(0, duration, 1 / samplerate)
plt.title("Noisy Time Domain")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.plot(time, data)
plt.show()

transform = fftpack.fft(data)
N = len(data)
n = np.arange(N)
freq = n*(samplerate/N)
mag = abs(transform)

#plt.xlim(0,12000)
plt.title("FFT Of Noisy Signal")
plt.xlabel("Freq")
plt.ylabel("Magnitude")
plt.plot(freq, mag)
plt.show()
winsound.PlaySound(r'C:\Users\adity\Desktop\course project\noisySYA13new (1).wav', winsound.SND_FILENAME)

#Filters
Fs=44100
nyq_rate =Fs/2
#Fc=450
#Fc1=Fc/Fs
N1 = 501
N2 = 750
N3 = 901
#a = signal.firwin(N1, cutoff = (1000/nyq_rate), window = "hamming", pass_zero="lowpass")
b = signal.firwin(N2, cutoff = (500/nyq_rate), window = "hanning", pass_zero=True)
c = signal.firwin(N3, cutoff = (230/nyq_rate,22049/nyq_rate), window = "blackmanharris", pass_zero="bandstop")
#d = signal.firwin(N, cutoff = (Fc1), window = "hanning", pass_zero='highpass')

#filtered_x = signal.lfilter(a, 1.0, data)
#filtered_x1 = signal.lfilter(b, 1.0, filtered_x)
filtered_x2 = signal.lfilter(c, 1.0, data)

#d1 = fftpack.fft(filtered_x)
#d2 = fftpack.fft(filtered_x1)
d3 = fftpack.fft(filtered_x2)

#m1 = abs(d1)
#m2 = abs(d2)
m3 = abs(d3)


plt.figure()
plt.xlim(0, 1000)
plt.title("FFT of Filtered signal")
plt.xlabel("Freq")
plt.ylabel("Magnitude")
plt.plot(freq,m3)
plt.show()

x2 = 10*filtered_x2
write("Filtered13.wav", samplerate, np.int16(x2))
winsound.PlaySound('Filtered13.wav', winsound.SND_FILENAME)


samplerate, data_filt = read("Filtered13.wav")
duration = len(data_filt) / samplerate
time = np.arange(0, duration, 1 / samplerate)

plt.title("Filtered Time Domain")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.plot(time, data_filt)
plt.show()


transform = np.fft.fft(data_filt)
N = len(data_filt)
n = np.arange(N)
freq = n*(samplerate/N)
mag = abs(transform)

plt.xlim(0,4000)
plt.title("FFT Of Filtered Signal")
plt.xlabel("Freq")
plt.ylabel("Magnitude")
plt.plot(freq, mag)
plt.show()