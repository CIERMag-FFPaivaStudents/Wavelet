#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is plot the FID/FFT graphs.
Only the Daubechies Family is used.
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

def plot_real(signal1,label1,signal2,label2,time,delta_time,title):
    size_y,size_x=1,2

    figure = plt.figure(figsize=(7,10))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')


    ax1 = figure.add_subplot(2,1,1)
    ax1.plot(time,signal1,'r-')
    ax1.set_ylim(-1,1)
    ax1.set_ylabel('Signal Amplitude (a.u.)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title('Real '+title)#,fontsize=23)



    ax1.tick_params(axis='x', labelsize=23)
    ax1.tick_params(axis='y', labelsize=23)

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
    offset = 25



    ax2.plot(xf,yplot,label = label1,color='#dd8452')


    ax2.set_ylabel('Spectral Amplitude (a.u.)',fontsize=26)
    ax2.set_xlabel('Frequency (kHz)',fontsize=26)


    ax2.tick_params(axis='x', labelsize=23)
    ax2.tick_params(axis='y', labelsize=23)

    ax2.set_ylim(-50,305)
    


    plt.tight_layout()
    


def plot(signal1,label1,signal2,label2,time,delta_time,title):
    size_y,size_x=1,2

    figure = plt.figure(figsize=(7,12))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')

    ax1 = figure.add_subplot(3,1,1)
    ax1.plot(time,np.real(signal1),'r-')
    ax1.set_ylim(-1,1)
    ax1.tick_params(axis='x', labelsize=23)
    ax1.tick_params(axis='y', labelsize=23)
    ax1.set_ylabel('Signal Amplitude (a.u.)',fontsize=26)
    ax1.set_xlabel('Time (ms)',fontsize=26)
    ax1.set_title('Real '+title)

    
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
    offset = 25

    ax2.plot(xf,yplot0+2*offset,label = label2)
    ax2.plot(xf,yplot+offset,label = label1)

    ax2.plot(xf,yplot0-yplot,label = 'Resíduo')
    ax2.tick_params(axis='x', labelsize=23)
    ax2.tick_params(axis='y', labelsize=23)
    ax2.set_ylabel('Spectral Amplitude (a.u.)',fontsize=26)
    ax2.set_xlabel('Frequency (kHz)',fontsize=26)



    plt.legend(loc='best',fontsize=23)
    ax2.set_ylim(-50,305)
    
    
    ax3 = figure.add_subplot(3,1,2)
    ax3.plot(time,np.imag(signal1),'m-')
    ax3.set_ylim(-1,1)
    ax3.set_ylabel('Signal Amplitude (a.u.)',fontsize=26)
    ax3.set_xlabel('Time (ms)',fontsize=26)
    ax3.set_title('Imaginary '+title)
    ax3.tick_params(axis='x', labelsize=23)
    ax3.tick_params(axis='y', labelsize=23)


    plt.tight_layout()
    
    plt.savefig(title+'.svg')
    plt.close()




def plot_fft(signal1,label1,signal2,label2,time,delta_time,title):

    figure = plt.figure(figsize=(10,7))
    
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

    ax2.plot(xf,yplot0+2*offset,label = label2)
    ax2.plot(xf,yplot+offset,label = label1)
    ax2.plot(xf,yplot0-yplot,label = 'Residue')

    ax2.set_ylabel('Spectral Amplitude (a.u.)',fontsize = 30)
    ax2.set_xlabel('Frequency (kHz)',fontsize = 30)
    

    ax2.tick_params(axis='x', labelsize=26)
    ax2.tick_params(axis='y', labelsize=26)

    
    plt.tick_params(axis='both', which='major')
    
    plt.legend(loc=2)
    ax2.set_ylim(-50,310)
    
    
    plt.title(title,fontsize=fs)
    plt.tight_layout()
    
    plt.savefig(title+'_fft.svg')
    plt.close()





if __name__ == '__main__':


    # print(os.getcwd())
    os.chdir('Data')
    os.chdir('..')

#Simulation input
    delta_time=0.5
    T2=100

    size_array=2048
    mean=0
    std_dev=0.1

    freq_list = [0,0.1,0.25,0.75]
    freq_title_list = ['000','010','025','075']
    
    
    freq=0.25
    title='Figura 12'
    signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
    
    signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)
    # plot_fft(signal_pure,'Sem ruído',signal_noise,'Ruidoso',time,delta_time,title)
    
    
    
    
    # plot(signal_pure,'Sem ruído',signal_noise,'Ruidoso',time,delta_time,title)
    
    
    # for freq in freq_list:
    for freq in [0.1]:
        print(freq)
        signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
    
        signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)
        


    #     # signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq)
    #     # signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev)
    #     os.chdir('Figures/FID_FFT')
    #     os.chdir(freq_title_list[freq_list.index(freq)])
    #     title='Noisy signal'
    #     # plot(signal_noise,'Noisy',signal_pure,'Noiseless',time,delta_time,title)

        title = '10a'
        print(title)
    #     # plot(signal_noise,'Noisy',signal_pure,'Noiseless',time,delta_time,title)
    #     plot(signal_noise,'Ruidoso',signal_pure,'Sem ruído',time,delta_time,title)
        plot(signal_pure,'Sem ruído',signal_noise,'Ruidoso',time,delta_time,title)
    #     # plt.clf()
    #     os.chdir('..')
    #     os.chdir('..')
    #     os.chdir('..')
    #     # os.chdir('..')
    #     # os.chdir('..')
        
    

    
    
        # bla = [8,9]
        # for num_params in range(1,25):
        # # for num_params in bla:
        #     print('\t%.d'%num_params)
        #     os.chdir('Data')
        #     params_name = 'Parameters'
        #     params = open(params_name+'.txt').readlines()[num_params]
        #     os.chdir('..')
        #     params = params.split('\t')
            
        #     noise_estimator, mode, wavelet, levels_dwt = params[1:]
        #     noise_estimator,levels_dwt = int(noise_estimator),int(levels_dwt)
    
    
      
        
        #     dir_name = 'FID_FFT/'
        #     os.chdir('Figures/'+dir_name)
        #     os.chdir(freq_title_list[freq_list.index(freq)])
        # #     alg = 'SURE'
        #     signal_smooth = filter_1D.Wavelet_filter(signal_noise,wavelet,levels_dwt,mode,'SURE')
            
            
        #     title='Smooth signal with sigma {} and params {}'.format(std_dev,num_params)
        # #     # title = 'fig 18_{}'.format(freq)
        #     # title = '10b'
        #     # plot(signal_smooth,'Smooth',signal_noise,'Noisy',time,delta_time,title)
        #     # plot(signal_smooth,'Filtrado',signal_noise,'Ruidoso',time,delta_time,title)
            
        #     os.chdir('FFT')
        #     # plot_fft(signal_smooth,'Smooth',signal_noise,'Noisy',time,delta_time,title)
        #     # plot_fft(signal_smooth,'Filtrado',signal_noise,'Ruidoso',time,delta_time,title)
        # #     # signal_smooth_real = filter_1D.Wavelet_filter(signal_noise_real,wavelet,levels_dwt,mode,alg)
        # #     # plot(signal_noise_real,'Noisy',signal_smooth_real,'Smooth',time,delta_time,title+' real',freq)
            
        # #     # plot(signal_noise_real,'Noisy',signal_smooth_real,'Smooth',time,delta_time,title+' real',freq)
        
        #     os.chdir('..')
        #     os.chdir('..')
        #     os.chdir('..')
        #     os.chdir('..')
            
            
            
            
            
    # for freq in [0.0]:
    # # for freq in freq_list:
    #     print(freq)
    #     signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
    
    #     signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)
        


        # title='fig10a'
        # # title='fig6a'

        # plot(signal_pure,'Sem ruído',signal_noise,'Ruidoso',time,delta_time,title)
        # plot_real(signal_noise,'Sem ruído',signal_noise,'Ruidoso',time,delta_time,title)
        
        # title='fig6c'
        # plot_real(signal_noise,'Sem ruído',signal_noise,'Ruidoso',time,delta_time,title)
        
        # title='fig11_{}'.format(100*freq)
        # plot_real(signal_noise,'Sem ruído',signal_noise,'Ruidoso',time,delta_time,title)

        # plot(signal_noise,'Sem ruído',signal_noise,'Ruidoso',time,delta_time,title)
        
        # bla = [8]
    #     for num_params in range(1,24):
    #     # for num_params in bla:
    #         print('\t%.d'%num_params)
    #         os.chdir('Data')
    #         params_name = 'Parameters'
    #         params = open(params_name+'.txt').readlines()[num_params]
    #         os.chdir('..')
    #         params = params.split('\t')
            
    #         noise_estimator, mode, wavelet, levels_dwt = params[1:]
    #         noise_estimator,levels_dwt = int(noise_estimator),int(levels_dwt)
    
    
    #         signal_smooth = filter_1D.Wavelet_filter(signal_noise,wavelet,levels_dwt,mode,'SURE')
            
            

    # #         title = 'fig10b'
    #         # title='teste'
    #         plot(signal_smooth,'Filtrado',signal_noise,'Ruidoso',time,delta_time,title)


    