#   Code for data generation, given scene parameters

import numpy as np
import matplotlib.pyplot as plt


plt.close('all')



#Scene Specifications

rad_dist = [30, 30, 30]               # Radial distance of target from radar in meters
rad_vel = [20, 50, 70]              # Radial velocity of target relative to radar in meter/sec



SNR = [30, 20, 40, 60]                   # SNRs at the detector for each of the target in dB



NoisePower = -120                # Noise Power in dB


# Chirp specification

fc = 79e9                        # Center frequency of chirp
BW = 2e9                         # Bandwidth of the chirp
Npts = 1024                      # Number of sample points of each chirp
Nramps = 512                      # Number of ramps
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


# Signal power  and amplitude calculation
sig_pow = np.zeros(len(SNR))
sig_amp = np.zeros(len(SNR))

for i in range(len(SNR)):
    sig_pow[i] = SNR[i] + NoisePower                            # Signal power in dB
    sig_amp[i] = np.sqrt(2* 10 ** (sig_pow[i]/10))

noise_pow = 10**(NoisePower/20)
AWGNnoise = np.random.normal(0, (noise_pow), Npts)



print("Range Resolution = ", RangeRes)
print("Range Maximum = ", RangeMax)
print("Velocity Resolution = ", VelRes)
print("Maximum Velocity = +-", VelMax)

ind = np.arange(Npts)

t = ind/fs

frame = np.zeros((1024, Nramps), dtype = 'complex64')
WindowRange = np.hanning(Npts)
WindowDoppler = np.hanning(Nramps)

ini_phase = 0+0j
for n in range(Nramps-1):
    IF_phase = np.zeros((Npts), dtype = 'complex64')
    phase = 0 + 0j
    for i in range(len(rad_dist)):

        f = (S * 2 * rad_dist[i]) / EM_vel
        # ini_phase = (4 * np.pi * fc * rad_dist[i]) / EM_vel
        IF_sig = (sig_amp[i] * np.sin((2 * np.pi * f * t) + (ini_phase))) + AWGNnoise                   
        phase = np.exp((1j * 4 * np.pi * fc * rad_vel[i] * (n) * Tramp)/EM_vel)        
        IF_phase += IF_sig * phase
        
    frame[:, n] = IF_phase * WindowRange                                                           # Windowing Each chirp

Range_FFT = np.fft.fft(frame, n = Npts, axis = 0) / Npts                                           # Range FFT of each chirp

frame = Range_FFT[:,:Nramps] * WindowDoppler[None, :]

aa = Range_FFT[:(Npts//2), :]                                                   # Windowing across chirps
Doppler_FFT = np.fft.fft(aa, n = Nramps, axis = 1) / Nramps            # Doppler FFT across chirps
# Doppler_FFTshift = np.fft.fftshift(Doppler_FFT[:(Npts//2), :], axes = 1)
plt.figure(1, dpi=200)
plt.plot(20*np.log10(abs(Range_FFT[:(Npts//2)])))
#plt.plot(abs(Range_FFT[:(Npts//2)]))
plt.title("Range FFT")
plt.xlabel("Range bins   ----------------->")
plt.ylabel("Power in dB    -------------->")
plt.grid(True)


plt.figure(2, dpi=200)
plt.plot(20*np.log10(np.abs(Doppler_FFT[267, :])))
plt.title("Doppler FFT")
plt.xlabel("Doppler bins    ------------->")
plt.ylabel("Power in dB    ------------->")
plt.grid(True)


# plt.figure(3)
# plt.plot(t,20*np.log10(AWGNnoise))

