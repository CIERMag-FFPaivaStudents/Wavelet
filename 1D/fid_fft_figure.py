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


def plot(signal1,label1,signal2,label2,time,delta_time,title,freq):
    size_y,size_x=1,2
    figure = plt.figure(figsize=(12*size_y,6*size_x))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')


    ax1 = figure.add_subplot(2,1,1)
    ax1.plot(time,np.real(signal1),'r-')
    ax1.set_ylim(-1,1)
    ax1.set_ylabel('Signal Amplitude (a.u.)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title(title,fontsize=23)

    size_array = len(signal1)
    yf=fft.fft(signal1)
    xf=fft.fftfreq(size_array,delta_time)
    xf=fft.fftshift(xf)
    yplot=np.abs((fft.fftshift(yf)))

    size_array = len(signal2)
    yf0=fft.fft(signal2)
    xf0=fft.fftfreq(size_array,delta_time)
    xf0=fft.fftshift(xf0)
    yplot0=np.abs((fft.fftshift(yf0)))

    ax2 = figure.add_subplot(2,1,2)
    offset = 50

    ax2.plot(xf,yplot0+2*offset,label = label2)
    ax2.plot(xf,yplot+offset,label = label1)
    ax2.plot(xf,yplot-yplot0,label = 'Residue')

    ax2.set_ylabel('Spectral Amplitude (a.u.)')
    ax2.set_xlabel('Frequency (kHz)')
    
    plt.legend(loc='best')
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
    # freq=0.25
    # freq_title = '025'
    size_array=2048
    mean=0
    std_dev=0.1

    freq_list = [0,0.1,0.25,0.75]
    freq_title_list = ['000','010','025','075']
    for freq in freq_list:
        signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
    
        signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)
        

        os.chdir('Figures/FID_FFT')
        os.chdir(freq_title_list[freq_list.index(freq)])
        title='Noisy signal'
        plot(signal_noise,'Noisy',signal_pure,'Noiseless',time,delta_time,title,freq)
    
        plt.clf()
        os.chdir('..')
        os.chdir('..')
        os.chdir('..')
        # os.chdir('..')
        # os.chdir('..')
    
        # signal_pure_real,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=True)
    
        # signal_noise_real = signal_1D.Add_Noise1D(signal_pure_real,mean,std_dev)
    
    
    
        for num_params in range(1,29):
    
            os.chdir('Data')
            params = open('Parameters.txt').readlines()[num_params]
            os.chdir('..')
            params = params.split('\t')
            
            noise_estimator, mode, wavelet, levels_dwt = params[1:]
            noise_estimator,levels_dwt = int(noise_estimator),int(levels_dwt)
    
    
      
        
        
            # os.chdir('Figures/FID_FFT/Paiva/freq '+freq_title)
            os.chdir('Figures/FID_FFT')
            os.chdir(freq_title_list[freq_list.index(freq)])
            alg = 'SURE'
            signal_smooth = filter_1D.Wavelet_filter(signal_noise,wavelet,levels_dwt,mode,'SURE')
            
            
            title='Smooth sigma {} and params {}'.format(std_dev,num_params)
            plot(signal_smooth,'Smooth',signal_noise,'Noisy',time,delta_time,title,freq)
            
            # signal_smooth_real = filter_1D.Wavelet_filter(signal_noise_real,wavelet,levels_dwt,mode,alg)
            # plot(signal_noise_real,'Noisy',signal_smooth_real,'Smooth',time,delta_time,title+' real',freq)
        
            os.chdir('..')
            os.chdir('..')
            os.chdir('..')
            # os.chdir('..')
            # os.chdir('..')
