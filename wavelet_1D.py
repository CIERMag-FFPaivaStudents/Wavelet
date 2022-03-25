
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to compute de Discrete Wavelet Transform of an MRI 1D signal(FID).
"""

import numpy as np
import matplotlib.pyplot as plt
import signal_1D
import pywt
import seaborn as sns

def DWT_1D(signal,wavelet,levels):
    aprox_coef0,detail_coef0=signal,0*signal
    approx_coef_list=[aprox_coef0]
    detail_coef_list=[detail_coef0]
    for i in range(levels):
        approx_coef,detail_coef=pywt.dwt(approx_coef_list[i],wavelet)
        approx_coef_list.append(approx_coef)
        detail_coef_list.append(detail_coef)
    return approx_coef_list,detail_coef_list


if __name__=='__main__':

#Simulation input
    delta_time=0.5
    T2=100
    freq=0.25 
    size_array=2048
    mean=0
    std_dev=0.01

#DWT input
    mother_wavelet='db'
    number_wavelet=4
    wavelet=mother_wavelet+'%d'%number_wavelet
    
    
    levels_dwt=3

    signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq)
    signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev)
    
    ca,cd = DWT_1D(signal_noise,wavelet,levels_dwt)
    
    

#Primeiro plot
    size_y,size_x=4,8
    figure = plt.figure(constrained_layout=False,figsize=(18*size_y,4*size_x))
    
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')
    
    gs = figure.add_gridspec(size_y, size_x)
    
    ax1=figure.add_subplot(gs[0, 0:])
    t_down= np.arange(0, size_array*delta_time, delta_time)
    ax1.plot(t_down,ca[0],'ko')
    ax1.set_title('Sinal com ruído',fontsize=40)
    ax1.set_ylabel('Amplitude do sinal (a.u.)',fontsize=30)
    ax1.set_xlabel('Tempo(ms)',fontsize=30)
    
    for level in range(1,levels_dwt+1):
        size_array_down = len(ca[level])
        t_down = np.arange(0, size_array*delta_time, delta_time*(size_array/size_array_down))
        
        ax_ca = figure.add_subplot(gs[level,0:2**(3-level)])
        ax_ca.plot(t_down,ca[level],'ko')
        ax_ca.set_title('{}º coeficiente de aproximação'.format(level),fontsize=40)
        ax_ca.set_ylabel('Amplitude do coeficiente (a.u.)',fontsize=30)
        ax_ca.set_xlabel('Tempo(ms)',fontsize=30)
        
        ax_cd = figure.add_subplot(gs[level,2**(3-level):2**(4-level)])
        ax_cd.plot(t_down,ca[level],'ko')
        ax_cd.set_title('{}º coeficiente de detalhe'.format(level),fontsize=40)
        ax_cd.set_ylabel('Amplitude do coeficiente (a.u.)',fontsize=30)
        ax_cd.set_xlabel('Tempo(ms)',fontsize=30)
        
        
    plt.tight_layout()
    plt.savefig('DWT1.jpeg')
    
    
# #Segundo plot
#     size_y,size_x=4,8
#     figure = plt.figure(constrained_layout=True,figsize=(12*size_y,4*size_x))
    
#     sns.set()
#     sns.set_style('ticks')
#     sns.set_context('talk')
    
#     gs = figure.add_gridspec(size_y, size_x)
    
#     ax1=figure.add_subplot(gs[0, 2:6])
#     t_down= np.arange(0, size_array*delta_time, delta_time)
#     ax1.plot(t_down,ca[0],'ko--')
#     ax1.set_title('Sinal com ruído')
#     ax1.set_ylabel('Amplitude do sinal (a.u.)')
#     ax1.set_xlabel('Tempo(ms)')
    
#     for level in range(1,levels_dwt+1):
#         size_array_down = len(ca[level])
#         t_down = np.arange(0, size_array*delta_time, delta_time*(size_array/size_array_down))
        
#         ax_ca = figure.add_subplot(gs[level,0:4])
#         ax_ca.plot(t_down,ca[level],'ko--')
#         ax_ca.set_title('{}º coeficiente de aproximação'.format(level))
#         ax_ca.set_ylabel('Amplitude do coeficiente (a.u.)')
#         ax_ca.set_xlabel('Tempo(ms)')
        
#         ax_cd = figure.add_subplot(gs[level,4:8])
#         ax_cd.plot(t_down,ca[level],'ko--')
#         ax_cd.set_title('{}º coeficiente de detalhe'.format(level))
#         ax_cd.set_ylabel('Amplitude do coeficiente (a.u.)')
#         ax_cd.set_xlabel('Tempo(ms)')


#     plt.tight_layout()
#     plt.savefig('DWT2.png')


