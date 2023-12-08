# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:33:20 2023

@author: admin
"""


import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

rad_dist = np.array(([5, 30, 20]), dtype = 'float64')               # Radial distance of target from radar in meters
rad_vel = np.array(([20, -20, 30]), dtype = 'float64')              # Radial velocity of target relative to radar in meter/sec


fc = 79e9                        # Center frequency of chirp
BW = 2e9                         # Bandwidth of the chirp
Npts = 1024                      # Number of sample points of each chirp
Nramps = 20                      # Number of ramps
fs = 112e6                       # Sampling frequency

EM_vel = 3e8                     # EM wave velocity in meter/sec


# Calculations of chirp and frame intervals

Tramp = Npts/fs
Tframe = Tramp * Nramps
S = BW/Tramp

# Calculation of range and velocity resolution

lamda = EM_vel / fc
RangeRes = EM_vel / (2*BW)                                                  # Range resolution
RangeMax = (EM_vel * fs) / (4 * S)                                               
VelRes = lamda / (2 * Tframe)                                               # Velocity resolution
VelMax = lamda / (4 * Tramp)


freq = (S * 2* rad_dist) / EM_vel

ind = np.arange(Npts)

t = ind/fs

sig = np.zeros((Npts), dtype = 'complex128')
for d in rad_dist:
    print(d)
    f = (S*2*d)/EM_vel
    sig += np.exp(1j * 2 * np.pi * f * t)

frame = np.zeros((Npts, Nramps), dtype = 'complex128')
# for ind in range(len(rad_vel)):
#     phase = np.exp((1j * 4 * np.pi * fc * rad_vel[ind] * (ind+1) * Tramp) / EM_vel)
#     range_fft = np.fft.fft(phase * sig)
#     frame[:,ind] = range_fft
    
range_fft = np.fft.fft(sig * np.hanning(1024))/1024
plt.plot(20*np.log10(np.abs(range_fft)))