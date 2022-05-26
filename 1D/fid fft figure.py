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
from scipy import fft


def plot(signal, signal_pure,time,delta_time,title,freq):
    size_y,size_x=1,2
    figure = plt.figure(figsize=(12*size_y,6*size_x))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')


    ax1 = figure.add_subplot(2,1,1)
    ax1.plot(time,np.real(signal),'r-')
    ax1.set_ylim(-1,1)
    ax1.set_ylabel('Signal Amplitude (a.u.)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title(title,fontsize=23)

    size_array = len(signal)
    yf=fft.fft(signal)
    xf=fft.fftfreq(size_array,delta_time)
    xf=fft.fftshift(xf)
    yplot=np.abs((fft.fftshift(yf)))

    size_array = len(signal_pure)
    yf0=fft.fft(signal_pure)
    xf0=fft.fftfreq(size_array,delta_time)
    xf0=fft.fftshift(xf0)
    yplot0=np.abs((fft.fftshift(yf0)))

    ax2 = figure.add_subplot(2,1,2)
    offset = 50

    ax2.plot(xf,yplot+2*offset,label = 'Noisy')
    ax2.plot(xf,yplot0+offset,label = 'Noiseless')
    ax2.plot(xf,yplot-yplot0,label = 'Residue')

    ax2.set_ylabel('Spectral Amplitude (a.u.)')
    ax2.set_xlabel('Frequency (kHz)')
    
    plt.legend(loc=1)
    ax2.set_ylim(-105,305)
    


    plt.tight_layout()
    
    plt.savefig(title+'.png')
    plt.close()

if __name__ == '__main__':


    # print(os.getcwd())
    os.chdir('Data')
    os.chdir('..')

#Simulation input
    delta_time=0.5
    T2=100
    freq=0.25
    size_array=2048
    mean=0
    std_dev=0.1

    signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)

    signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev)
    
    os.chdir('Figures/FID_FFT')
    title='Noisy signal'
    plot(signal_noise,signal_pure,time,delta_time,title,freq)

    plt.clf()
    os.chdir('..')
    os.chdir('..')

    for num_params in range(1,25):

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
    

    
    
    
        os.chdir('Figures/FID_FFT')
        
        alg = 'SURE'
        signal_smooth = filter_1D.Wavelet_filter(signal_noise,wavelet,levels_dwt,mode,alg)
        
        
        title='Smooth sigma {} and params {}'.format(std_dev,num_params)
        plot(signal_smooth, signal_pure,time,delta_time,title,freq)
    
        os.chdir('..')
        os.chdir('..')