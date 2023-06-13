# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np

BW = 1000
plt.close('all')

S_slope = 100
T_c = BW/S_slope
fs = 16000

t = np.arange(0, T_c, 1/fs)

#f=np.linspace(0, BW, N)

pi_t = 2* np.pi* 0.5 * S_slope* pow(t,2)

x_t = 0.1*np.sin(pi_t)

plt.figure(1),plt.plot(t,x_t)
plt.grid(True)


freq = np.fft.fftfreq(len(x_t), 1/(fs))
#freq = freq[:(len(freq)-1)]
#freq = np.arange(-1*BW, BW, 1)
X_f = np.fft.fft(x_t)
# X_f = np.fft.fftshift(X_f)
# plt.figure(4),plt.plot(20*np.log10(abs(X_f))-np.amax(20*np.log10(abs(X_f))))
# plt.grid(True)
# #plt.gca().xaxis.set_ticklabels(2*t)
X_f[int(len(x_t)/2)] = 0
plt.figure(2),plt.plot(freq, 20*np.log10(abs(X_f)))
plt.grid(True)


plt.figure(3),plt.plot(freq, np.angle(X_f))
plt.grid(True)

xt = np.fft.ifft(X_f)
plt.figure(4),plt.plot(t, xt)
plt.grid(True)




