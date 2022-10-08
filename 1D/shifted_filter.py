#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is plot the FID/FFT graphs for different mother wavelets and frequency shifted signal.
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
import fid_fft_figure as fff
def plot_fft(signal1,label1,signal2,label2,time,delta_time,title,freq):


    figure = plt.figure(figsize=(9,6))
    fs=25

    sns.set()
    sns.set_context('talk')
    sns.set_style('white')


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

    ax2 = figure.add_subplot(1,1,1)
    offset = 50


    ax2.plot(xf,yplot,label = label1,color='#dd8452')


    peak = np.where(yplot0==yplot0.max())
    ax2.plot(xf0[peak],yplot0[peak],'rX',ms=20)

    
    plt.tick_params(axis='both', which='major')
    
    ax2.set_ylim(-105,305)
    
    
    plt.title(title,fontsize=fs)
    plt.tight_layout()
    
    plt.savefig(title+'_fft.png')
    plt.close()



def plot(signal1,label1,signal2,label2,time,delta_time,title,freq):
    size_y,size_x=1,2
    figure = plt.figure(figsize=(18*size_y,10*size_x))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')


    ax1 = figure.add_subplot(3,1,1)
    ax1.plot(time,np.real(signal1),'r-')
    ax1.set_ylim(-1,1)
    ax1.set_ylabel('Signal Amplitude (a.u.)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title('Real '+title,fontsize=23)

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

    ax2 = figure.add_subplot(3,1,3)
    offset = 50

    ax2.plot(xf,yplot0+2*offset,label = label2)
    ax2.plot(xf,yplot+offset,label = label1)
    ax2.plot(xf,yplot-yplot0,label = 'Residue')

    peak = np.where(yplot0==yplot0.max())
    ax2.plot(xf0[peak],yplot0[peak]+2*offset,'r*',ms=20)

    ax2.set_ylabel('Spectral Amplitude (a.u.)')
    ax2.set_xlabel('Frequency (kHz)')
    
    plt.legend(loc='best')
    ax2.set_ylim(-105,305)
    
    ax3 = figure.add_subplot(3,1,2)
    ax3.plot(time,np.imag(signal1),'m-')
    ax3.set_ylim(-1,1)
    ax3.set_ylabel('Signal Amplitude (a.u.)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_title('Imaginary '+title,fontsize=23)


    plt.tight_layout()
    
    plt.savefig(title+'.png')
    plt.close()

def shift_freq(signal,time,delta_time):

    size_array = len(signal)
    yf0=fft.fft(signal)
    xf0=fft.fftfreq(size_array,delta_time)
    xf0=fft.fftshift(xf0)
    yplot0=np.abs((fft.fftshift(yf0)))

    peak = np.where(yplot0==yplot0.max())
    freq = xf0[peak][0]

    shifted_signal = signal*np.exp(-1j*2*np.pi*freq*time)

    return shifted_signal,freq
    

if __name__ == '__main__':


    os.chdir('Data')
    os.chdir('..')

#Simulation input
    delta_time=0.5
    T2=100
    size_array=2048
    mean=0
    std_dev=0.1


    
    




    for num_params in range(25,49):

        os.chdir('Data')
        params_name = 'Parameters2'
        params = open(params_name+'.txt').readlines()[num_params-24]
        os.chdir('..')
        
        params = params.split('\t')


        freq,mode, wavelet, levels_dwt = params[1:]
        freq,levels_dwt = float(freq),int(levels_dwt)

        signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
    
        signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev,False)


        dir_name = 'FID_FFT4'
        # os.chdir('Figures/'+dir_name)


        
        # title='Noisy_signal_{}kHz'.format(freq)
        # title='coisa 250hz'
        # fff.plot(signal_noise,'Noisy',signal_pure,'Noiseless',time,delta_time,title)
        # plot_fft(signal_noise,'Noisy',signal_pure,'Noiseless',time,delta_time,title,freq)
        # plt.clf()
        # os.chdir('..')
        # os.chdir('..')

        
        shifted_signal,freq_peak = shift_freq(signal_noise, time, delta_time)
        
        os.chdir('Figures/'+dir_name)

        signal_smooth = filter_1D.Wavelet_filter(shifted_signal,wavelet,levels_dwt,mode,'SURE')
    
        signal_smooth_reshift = signal_smooth*np.exp(1j*2*np.pi*freq_peak*time)
                    
        title='Smooth signal with sigma {} and params {}'.format(std_dev,num_params)
        # fff.plot(signal_smooth_reshift,'Smooth_reshift',signal_noise,'Noisy',time,delta_time,title)
        # print(os.getcwd())
        # fff.plot(signal_smooth_reshift,'Filtrado',signal_noise,'Ruidoso',time,delta_time,title)
         
        os.chdir('FFT')
        
        # fff.plot_fft(signal_smooth_reshift,'Smooth_reshift',signal_noise,'Noisy',time,delta_time,title)
        fff.plot_fft(signal_smooth_reshift,'Filtrado',signal_noise,'Ruidoso',time,delta_time,title)


        os.chdir('..')
        os.chdir('..')
        os.chdir('..')
