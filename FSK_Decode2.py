import numpy as np
import pylab as pl
import scipy.signal.signaltools as sigtool
import scipy.signal as signal
from numpy.random import sample
from sklearn.cluster import KMeans
import wave
# import matplotlib.pyplot as plt
# the following variables setup the system
Fc = 10000       # simulate a carrier frequency of 1kHz
Fdev = 2500      # frequency deviation, make higher than bitrate

Fbit = 300     # simulated bitrate of data
gapRatio = 0.5

A = 1           # transmitted signal amplitude
Fs = 44100      # sampling frequency for the simulator, must be higher than twice the carrier frequency

data_in = []
realN = 1024
sequence = [1, 0, 1, 0, 0, 1, 0, 1]
N = 0        # how many bits to send

'''
使用KMeans算法一维聚类，请传入一个np数组s
当前默认为三个类
'''
def KMeans_Alg(s, n_cluster = 3):
    s = s.reshape(-1, 1) # 确保为一维数组
    km = KMeans(n_cluster)
    km.fit(s)
    print('聚类后的列表')
    print(km.cluster_centers_)
    fine = list(sorted(km.cluster_centers_))[1:] # 剔除最小类
    # print(fine)
    return np.mean(fine)

def setDataIn(s):
    global data_in, realN, N
    realN = len(s)
    padding = int(Fbit * gapRatio)
    #N = int(realN / ratio) + 2 * len(sequence) + start        # how many bits to send
    N = realN + 2 * len(sequence) + 2 * padding

    data_in = np.hstack((np.array([0] * padding + sequence), s, np.array(sequence)))
    data_in = np.hstack((data_in, np.array([0] * (N - len(data_in)))))
    #data_in = list(data_in)

def generateFile(path, data):
    t = np.arange(0,N/Fbit,1/Fs, dtype=float)
    m = np.zeros(0).astype(float)
    for bit in data:
        if bit == 0:
            m=np.hstack((m,np.multiply(np.ones(Fs//Fbit),Fc+Fdev)))
        else:
            m=np.hstack((m,np.multiply(np.ones(Fs//Fbit),Fc-Fdev)))
    y0=A * np.cos(2*np.pi*np.multiply(m,t))

    f = wave.open(path, "wb")
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(44100)
    y0 = (y0 * 32768).astype(np.int16)
    f.writeframes(y0.tobytes())
    f.close()

def str2bin(s):
    t = list(s)
    r = []
    for i in t:
        for j in bin(ord(i))[2:].rjust(8,'0'):
            r.append(int(j))
    return r

def bin2str(s):
    t = ''.join([str(i) for i in s])
    r = ''
    for i in range(0, len(t), 8):
        r += chr(int(t[i: i + 8], 2))
    return r

def index(src, dst = sequence, start = 0):
    s = ''.join([str(i) for i in src])
    d = ''.join([str(i) for i in dst])
    if d in s[start:]: return s[start:].index(d)
    else: return -1

def lastindex(src, dst = sequence, start = 0):
    s = ''.join([str(i) for i in src])
    d = ''.join([str(i) for i in dst])
    if d in s[start:]: return s[start:].rindex(d)
    else: return -1

def check(src, dst):
    i = 0
    for m, n in zip(src, dst):
        if m == n: i += 1
    print("Correct bits: " + str(i) + "/" + str(len(src)))

def getData(path, mean = 0):
    f = wave.open(path, "rb")
    y = f.readframes(f.getnframes())
    f.close()
    y = np.frombuffer(y, dtype=np.int16)
    y = y / 32768

    #Differentiator
    y_diff = np.diff(y,1)
    #Envelope detector + low-pass filter
    # create an envelope detector and then low-pass filter
    y_env = np.abs(sigtool.hilbert(y_diff))
    h=signal.firwin( numtaps=100, cutoff=Fbit*2, nyq=Fs/2)
    y_filtered=signal.lfilter( h, 1.0, y_env)
    #slicer
    rx_data = []
    sampled_signal = y_filtered[int(Fs/Fbit/2):len(y_filtered):Fs//Fbit]

    # calculate the mean of the signal
    # modified
    # if the mean of the bit period is higher than the mean, the data is a 0
    if mean == 0:
        #mean = KMeans_Alg(y_filtered) # 传入更大的数组来聚类
        mean = KMeans_Alg(sampled_signal)
    print(mean)

    s = list(sorted(sampled_signal))
    pl.scatter(range(len(s)), s)
    s = [s[i] - s[i - 1] for i in range(1, len(s))]
    pl.scatter(range(len(s)), s)
    pl.show()

    pl.scatter(list(range(len(sampled_signal))), sampled_signal,s = 1)
    pl.show()

    for bit in sampled_signal:
        if bit > mean:
            rx_data.append(0)
        else:
            rx_data.append(1)
    rx_data = rx_data[rx_data.index(0):]
    while rx_data[-1] == 1: rx_data = rx_data[:-1]

    head = index(rx_data)
    if 0 < head <= Fbit * gapRatio:
        print("识别到头部特征串")
        rx_data = rx_data[head + len(sequence):]
        tail = lastindex(rx_data)
        if tail >= realN:
            print("识别到尾部特征串")
            rx_data = rx_data[: tail]
            if tail > realN:
                print("Length error.")
        else:
            print("未识别到尾部特征串")
            print(list(data_in))
            print(rx_data)
    else:
        print("未识别到头部特征串")
        print(list(data_in))
        print(rx_data)
    '''
    tail = lastindex(rx_data)
    if head == -1 or tail - head - len(sequence) < realN:
        print("未识别到特征串")
        print(list(data_in))
        print(rx_data)
    elif tail - head - len(sequence) == realN:
        rx_data = rx_data[head + len(sequence): tail]
        print("特征串完美识别")
    else:
        rx_data = rx_data[head + len(sequence): tail]
        print("未能正确识别特征串")
    '''
    return rx_data


src = 'HelloWorld ' * 12
src = \
'''
   <xmpDM:altTimecode
    xmpDM:timeValue="00:00:00:00"
    xmpDM:timeFormat="25Timecode"/>
'''
src = '''import numpy as np
import pylab as pl
import scipy.signal.signaltools as sigtool
import scipy.signal as signal
from numpy.random import sample
from sklearn.cluster import KMeans
import wave'''

setDataIn(str2bin(src))
command = int(input("1.Generate file\n2.Analysis output.\n"))
if command == 1:
    generateFile("FSK.wav", data_in)
elif command == 2:
    #dst = bin2str(getData('test.wav'))
    dst = bin2str(getData('test_output.wav'))
    print('=' * 32)
    print("src:" + src)
    print('=' * 32)
    print("dst:" + dst)
    print('=' * 32)
    print(src == dst[: len(src)])
    check(str2bin(src), str2bin(dst[:len(src)]))

'''
#print(rx_data)
src = getData('test.wav')
dst = getData('test.output.wav', 0.02)
#print(src)
print()
#print(dst)
'''