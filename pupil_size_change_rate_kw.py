import numpy as np
import pandas as pd
from scipy import fftpack
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from interpolation import *


from numpy import arange, ones, pi
from scipy.fftpack import fft, fftfreq, ifft



# def numpy_zero_to_nan(input):
#
#     while 0 in y:
#         y.remove(0)
#
#     return output


# 신호를 생성합니다.
# Seed the random number generator
n_frame_st = 10000
n_frame = 60000

csv = pd.read_csv('./csvs_final/전태식가.csv')
## tolist()를 통해서 [''] 특정열의 객체를 list 로 변경
y = csv['pupil_size_diameter'].tolist()
y = np.array(y)
print(y.size)

# data = np.array(y)
# print(data.dtype)
# print(data[1050:1100])
# deldata = np.delete(data, 0)
# print(deldata[1050:1100])

# while 0 in y:
#     y.remove(0)

y = np.array(y[n_frame_st:n_frame], dtype=np.float64)
ori_y = y
x = np.arange(n_frame_st, n_frame)

print(y[0:20])
y = list_zero_interpolation(y, len(x))
# print(y[:20])

# print(y)
new_y = y
y = np.array(y, dtype=np.int32)
print(y[:20])
# print(ori_y)
# print(y)
# print(new_y)

# print(y[2843:2870])
# FFT의 Power를 계산합니다.
# The FFT of the signal
### 입력 배열에 대해 fft 계산
sig_fft = fftpack.fft(y)
# And the power (sig_fft is of complex dtype)
power = np.abs(y)
# The corresponding frequencies
# sample_freq = fftpack.fftfreq(y.size, d=time_step)
print(y.size)

### 임의로 y size의 진동수를 만들어준다.
sample_freq = fftpack.fftfreq(y.size, 0.001)
# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

# Check that it does indeed correspond to the frequency that we generate
# the signal with\

# 모든 high frequencies를 제거합니다.
high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)
print(y[43200:44250])
plt.figure(figsize=(6, 5))
plt.subplot(211)
# plt.plot(x, ori_y, label='Original signal')
plt.plot(x, y, linewidth=3, label='new_y', color='pink')
plt.plot(x, filtered_sig, linewidth=1, label='Filtered signal')
plt.legend(loc='best')
plt.xlim(43100, 44250)
plt.ylim(0, 110)

plt.subplot(212)
plt.plot(x, y, linewidth=1, label='power')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')
plt.ylim(0,110)
plt.xlim(43100, 44250)


# plt.subplot(313)

# plt.plot(x, power, label='power')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.xlim(2843,2870)

plt.show()

# plt.figure(figsize=(6, 5))
# plt.plot(x, ori_y, label='test')
# plt.plot(x, new_y, label='new')
# plt.xlabel('time')
# plt.ylabel('Amplitude')
#
# plt.legend(loc='best')
#
# plt.show()


