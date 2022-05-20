#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is plot the FID/FFT graphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import signal_1D
import wavelet_1D
import sigmas
import metrics_1D
import filter_1D
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import seaborn as sns
from scipy import fftpack


def plot(FID,time,delta_time,title):
    size_y,size_x=1,2
    figure = plt.figure(figsize=(12*size_y,6*size_x))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')



    ax1 = figure.add_subplot(2,1,1)
    ax1.plot(time,FID,'r-')
    ax1.set_ylim(-1,1)
    ax1.set_ylabel('Signal Amplitude (a.u.)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title(title,fontsize=23)

    size_array = len(FID)
    yf=fftpack.fft(FID)
    xf=fftpack.fftfreq(size_array,delta_time) 
    xf=fftpack.fftshift(xf)
    Xplot=np.abs((fftpack.fftshift(yf)))*delta_time

    ax2 = figure.add_subplot(2,1,2)
    ax2.plot(xf, Xplot,'b-')
    ax2.set_xlim(-0,1)
    ax2.set_ylabel('Spectral Amplitude (a.u.)')
    ax2.set_xlabel('Frequency (kHz)')
    
    
    plt.tight_layout()
    
    plt.savefig(title+'.png')
    

if __name__ == '__main__':


    for num_params in range(1,25):
        # num_params = 13+i
        os.chdir('Data')
        params = open('Parameters.txt').readlines()[num_params]
        os.chdir('..')
        params = params.split('\t')
        noise_estimator, mode, wavelet, levels_dwt = params[1:]
        noise_estimator,levels_dwt = int(noise_estimator),int(levels_dwt)
    
    #Simulation input
        delta_time=0.5
        T2=100
        freq=0.25
        size_array=2048
        mean=0
        std_dev=0.1
    
        signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq)
        signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev)
        
        ca,cd = wavelet_1D.DWT_1D(signal_noise,wavelet,levels_dwt)
    
    
    
        os.chdir('Figures/FID_FFT')
        # title='sem ruído'
        # plot(signal_pure,time,delta_time,title)
    
        # title='sinal ruidoso sigma {}'.format(std_dev)
        # plot(signal_noise,time,delta_time,title)
        
        
        cd_smooth = filter_1D.Adaptive_smoothing(signal_pure,cd,mode)
        signal_smooth = filter_1D.IDWT_1D(ca[-1],cd_smooth,wavelet,levels_dwt)
        
        
        title='suave sigma {} e params {}'.format(std_dev,num_params)
        plot(signal_smooth,time,delta_time,title)
    
        os.chdir('..')
        os.chdir('..')
