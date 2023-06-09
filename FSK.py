import numpy as np
import pylab as pl
import scipy.signal.signaltools as sigtool
import scipy.signal as signal
from numpy.random import sample
import wave
# import matplotlib.pyplot as plt

#the following variables setup the system
Fc = 1000       #simulate a carrier frequency of 1kHz
Fbit = 50       #simulated bitrate of data
Fdev = 500      #frequency deviation, make higher than bitrate
realN = 1024
ratio = 0.96
sequence = [1, 0, 1, 0, 0, 1, 0, 1]
N = int(realN / 0.96) + 2 * len(sequence)        #how many bits to send
start = (N - realN) // 2 - len(sequence)

A = 1           #transmitted signal amplitude
Fs = 10000      #sampling frequency for the simulator, must be higher than twice the carrier frequency
A_n = 0.2      #noise peak amplitude
N_printbits = N // 4 #number of bits to print in plots

def plot_data(y):
    #view the data in time and frequency domain
    #calculate the frequency domain for viewing purposes
    N_FFT = float(len(y))
    f = np.arange(0,Fs/2,Fs/N_FFT)
    w = np.hanning(len(y))
    y_f = np.fft.fft(np.multiply(y,w))
    y_f = 10*np.log10(np.abs(y_f[0:int(N_FFT/2)]/N_FFT))
    point_num = int((Fc+Fdev*2)*N_FFT/Fs)
    pl.subplot(3,1,1)
    pl.plot(t[0:Fs*N_printbits//Fbit],m[0:Fs*N_printbits//Fbit])
    pl.xlabel('Time (s)')
    pl.ylabel('Frequency (Hz)')
    pl.title('Original VCO output versus time')
    pl.grid(True)
    pl.subplot(3,1,2)
    pl.plot(t[0:Fs*N_printbits//Fbit],y[0:Fs*N_printbits//Fbit])
    pl.xlabel('Time (s)')
    pl.ylabel('Amplitude (V)')
    pl.title('Amplitude of carrier versus time')
    pl.grid(True)
    pl.subplot(3,1,3)
    pl.plot(f[0:point_num],y_f[0:point_num])
    pl.xlabel('Frequency (Hz)')
    pl.ylabel('Amplitude (dB)')
    pl.title('Spectrum')
    pl.grid(True)
    pl.tight_layout()
    #pl.show()
    
"""
Data in
"""
#generate some random data for testing
#data_in = [0, 1] * (realN // 2)
data_in = np.random.randint(0,2,realN)

data_in = np.hstack((np.array([0] * start + sequence), data_in, np.array(sequence)))
data_in = np.hstack((data_in, np.array([0] * (N - len(data_in)))))

#print(data_in)
"""
VCO
"""
t = np.arange(0,N/Fbit,1/Fs, dtype=float)
#extend the data_in to account for the bitrate and convert 0/1 to frequency
m = np.zeros(0).astype(float)
for bit in data_in:
    if bit == 0:
        m=np.hstack((m,np.multiply(np.ones(Fs//Fbit),Fc+Fdev)))
    else:
        m=np.hstack((m,np.multiply(np.ones(Fs//Fbit),Fc-Fdev)))
#calculate the output of the VCO
y=np.zeros(0)
y0=A * np.cos(2*np.pi*np.multiply(m,t))

# plt.scatter(range(0,len(y)),y)

plot_data(y0)

# 音频产生
f = wave.open(r"FSK.wav", "wb")

# 配置声道数、量化位数和取样频率
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(44100)
# 将wav_data转换为二进制数据写入文件
y0 = (y0 * 32768).astype(np.int16)
f.writeframes(y0.tobytes())
#print(y0.tobytes())
f.close()



"""
Noisy Channel0
"""
'''
#create some noise
noise = (np.random.randn(len(y))+1)*A_n
snr = 10*np.log10(np.mean(np.square(y)) / np.mean(np.square(noise)))
print("SNR = %fdB" % snr)
y=np.add(y,noise)
#view the data after adding noise
'''
f = wave.open(r"FSK.wav", "rb")
y = f.readframes(f.getnframes())
f.close()

#print(y0.tobytes() == y)
y = np.frombuffer(y, dtype=np.int16)
y = y / 32768

plot_data(y)

"""
Differentiator
"""
y_diff = np.diff(y,1)

"""
Envelope detector + low-pass filter
"""
#create an envelope detector and then low-pass filter
y_env = np.abs(sigtool.hilbert(y_diff))
h=signal.firwin( numtaps=100, cutoff=Fbit*2, nyq=Fs/2)
y_filtered=signal.lfilter( h, 1.0, y_env)
#view the data after adding noise
N_FFT = float(len(y_filtered))
f = np.arange(0,Fs/2,Fs/N_FFT)
w = np.hanning(len(y_filtered))
y_f = np.fft.fft(np.multiply(y_filtered,w))
y_f = 10*np.log10(np.abs(y_f[0:int(N_FFT/2)]/N_FFT))
point_num = int((Fc+Fdev*2)*N_FFT/Fs)
pl.subplot(3,1,1)
pl.plot(t[0:Fs*N_printbits//Fbit],m[0:Fs*N_printbits//Fbit])
pl.xlabel('Time (s)')
pl.ylabel('Frequency (Hz)')
pl.title('Original VCO output vs. time')
pl.grid(True)
pl.subplot(3,1,2)
pl.plot(t[0:Fs*N_printbits//Fbit],np.abs(y[0:Fs*N_printbits//Fbit]),'b')
pl.plot(t[0:Fs*N_printbits//Fbit],y_filtered[0:Fs*N_printbits//Fbit],'g',linewidth=3.0)
pl.xlabel('Time (s)')
pl.ylabel('Amplitude (V)')
pl.title('Filtered signal and unfiltered signal vs. time')
pl.grid(True)
pl.subplot(3,1,3)
pl.plot(f[0:point_num],y_f[0:point_num])
pl.xlabel('Frequency (Hz)')
pl.ylabel('Amplitude (dB)')
pl.title('Spectrum')
pl.grid(True)
pl.tight_layout()
pl.show()

"""
slicer
"""
#calculate the mean of the signal
mean = np.mean(y_filtered)
#if the mean of the bit period is higher than the mean, the data is a 0
rx_data = []
sampled_signal = y_filtered[int(Fs/Fbit/2):len(y_filtered):Fs//Fbit]
for bit in sampled_signal:
    if bit > mean:
        rx_data.append(0)
    else:
        rx_data.append(1)

bit_error=0
for i in range(0,len(data_in)):
    if rx_data[i] != data_in[i]:
        print(i)
        bit_error+=1
print("bit errors = %d" % bit_error)
print("bit error percent = %4.2f%%" % (bit_error/N*100))
