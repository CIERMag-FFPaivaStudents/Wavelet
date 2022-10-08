
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to compute de Discrete Wavelet Transform of an MRI 1D signal(FID) and generate figures to illustrate it.
"""

import numpy as np
import matplotlib.pyplot as plt
import signal_1D
import pywt
import seaborn as sns
import os

def DWT_1D(signal,wavelet,levels):
    """Calculates the Discrete Wavelet Transform of an 1D magnetic ressonance signal.
    Parameters
    ----------
    signal: array
        Magnetic ressonance free induction decay signal.
        
    wavelet: Wavelet object or name string
        Desired mother wavelet in the DWT.
        
    levels: int
        Number of decomposition levels.
        
    Return
    ------
    approx_coef_list: array
        List of approximation coefficients from the lowest level to the highest.

    detail_coef_list: array
        List of detail coefficients from the lowest level to the highest.

    References
    ----------
        
    """
    
    aprox_coef0,detail_coef0=signal,0*signal
    
    approx_coef_list= (levels+1)*[None]
    detail_coef_list = (levels+1)*[None]
    
    approx_coef_list[0] = aprox_coef0
    detail_coef_list[0] = detail_coef0
    for level in range(levels):
        approx_coef,detail_coef=pywt.dwt(approx_coef_list[level],wavelet)
        approx_coef_list[level+1] = approx_coef
        detail_coef_list[level+1] = detail_coef
    return approx_coef_list,detail_coef_list


def First_figure_parts(name,size_array,delta_time,levels_dwt,ca,cd):
    # size_y,size_x=1,12
    plt.figure(figsize=(18,8))
    
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')
    


    
    t_down= np.arange(0, size_array*delta_time, delta_time)
    plt.plot(t_down,ca[0],'ko-',ms=4,lw=1)
    plt.ylim(-2.5,2.5)
    # plt.title('Noisy signal')

    # plt.ylabel('Amplitude (a.u.)')
    # plt.xlabel('Time(ms)')
    plt.xticks([])
    plt.yticks([])
    os.chdir('Figures')
    plt.tight_layout()
    plt.savefig(name+'_noisy signal.png')

    plt.clf()
    plt.close()

    for level in range(1,levels_dwt+1):
        
        size_array_down = len(ca[level])
        t_down = np.arange(0, size_array*delta_time, delta_time*(size_array/size_array_down))
        
        plt.figure(figsize=(18,8))
        
        sns.set()
        sns.set_style('ticks')
        sns.set_context('talk')
        
        plt.plot(t_down,ca[level],'ko-',ms=4,lw=1)
        # plt.title('ca{}'.format(level))
        plt.xticks([])
        plt.yticks([])
        # plt.ylabel('Amplitude (a.u.)')
        # plt.xlabel('Time(ms)')
        plt.ylim(-2.5,2.5)
        plt.tight_layout()
        plt.savefig(name+'_ca{}'.format(level)+'.png')

        plt.clf()
        plt.close()

        plt.figure(figsize=(18,8))
        
        sns.set()
        sns.set_style('ticks')
        sns.set_context('talk')

        plt.plot(t_down,cd[level],'ko-',ms=4,lw=1)

        # plt.title('cd{}'.format(level))

        # plt.ylabel('Amplitude (u.a.)')
        plt.ylim(-2.5,2.5)
        plt.xticks([])
        plt.yticks([])
        # plt.xlabel('Time(ms)')
        plt.tight_layout()
        plt.savefig(name+'_cd{}'.format(level)+'.png')

        plt.clf()
        plt.close()
    os.chdir('..')

    return None

def First_figure(name,size_array,delta_time,levels_dwt,ca,cd):
    """Shows a figure of wavelet coefficients, emphasizing the difference in size.
    Parameters
    ----------
    name: string
        Desired name of the image.
        
    size_array: int
        Desired length of the simulated 1D signal.
        
    delta_time: float
        Interval between consecutive points of time of the simulated 1D signal.
        
    levels_dwt: int
        Number of DWT decomposition levels.
        
    ca: array
        List of approximation coefficients from the lowest level to the highest.

    cd: array
        List of detail coefficients from the lowest level to the highest.
        
    Return
    ------

    References
    ----------
    """
    size_y,size_x=8,25
    figure = plt.figure(figsize=(size_y,size_x))
    
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')
    
    gs = figure.add_gridspec(size_y, size_x)
    
    fs=20
    
    ax1=figure.add_subplot(gs[0, 0:])
    t_down= np.arange(0, size_array*delta_time, delta_time)
    ax1.plot(t_down,ca[0],'ko-',ms=1)
    ax1.set_title('Noisy signal',fontsize=fs)

    ax1.set_ylabel('Signal amplitude (a.u.)',fontsize=fs)

    ax1.set_xlabel('Time (ms)',fontsize=fs)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim(-2.5,2.5)
    for level in range(1,levels_dwt+1):
        size_array_down = len(ca[level])
        t_down = np.arange(0, size_array*delta_time, delta_time*(size_array/size_array_down))
        
        ax_ca = figure.add_subplot(gs[level,0:2**(3-level)])
        ax_ca.plot(t_down,ca[level],'ko-')
        ax_ca.set_title('ca{}'.format(level),fontsize=fs)

        # ax_ca.set_ylabel('Amplitude of coefficient (a.u.)',fontsize=fs)
        ax_ca.set_ylabel('Amplitude',fontsize=fs)
        ax_ca.set_xlabel('Time',fontsize=fs)
        ax_ca.set_xticks([])
        ax_ca.set_yticks([])
        ax_ca.set_ylim(-2.5,2.5)

        ax_cd = figure.add_subplot(gs[level,2**(3-level):2**(4-level)])
        ax_cd.plot(t_down,cd[level],'ko-')
        ax_cd.set_title('cd{}'.format(level),fontsize=fs)
        ax_cd.set_xlabel('Time',fontsize=fs)
        ax_cd.set_xticks([])
        ax_cd.set_yticks([])
        ax_cd.set_ylim(-2.5,2.5)
        
    os.chdir('Figures')
    plt.tight_layout()
    plt.savefig(name+'.png')
    os.chdir('..')
    return None
    
    
def Second_figure(name,size_array,delta_time,levels_dwt,ca,cd):
    """Shows a figure of wavelet coefficients with the same size.
    Parameters
    ----------
    name: string
        Desired name of the image.
        
    size_array: int
        Desired length of the simulated 1D signal.
        
    delta_time: float
        Interval between consecutive points of time of the simulated 1D signal.
        
    levels_dwt: int
        Number of DWT decomposition levels.
        
    ca: array
        List of approximation coefficients from the lowest level to the highest.

    cd: array
        List of detail coefficients from the lowest level to the highest.
        
    Return
    ------

    References
    ----------
    """
    size_y,size_x=1,2
    figure = plt.figure(constrained_layout=True,figsize=(12*size_y,4*size_x))


    sns.set()
    sns.set_context('talk')
    sns.set_style('white')
    
    gs = figure.add_gridspec(size_y, size_x)
    
    ax1=figure.add_subplot(gs[0, 2:6])
    t_down= np.arange(0, size_array*delta_time, delta_time)
    ax1.plot(t_down,ca[0],'ko--')
    ax1.set_title('Noisy signal')
    ax1.set_ylabel('Signal amplitude')#' (a.u.)')
    ax1.set_xlabel('Time')#'(ms)')
    
    for level in range(1,levels_dwt+1):
        size_array_down = len(ca[level])
        t_down = np.arange(0, size_array*delta_time, delta_time*(size_array/size_array_down))
        
        ax_ca = figure.add_subplot(gs[level,0:4])
        ax_ca.plot(t_down,ca[level],'ko--')
        ax_ca.set_title('{}º approximation coefficient'.format(level))
        ax_ca.set_ylabel('Amplitude of coefficient (a.u.)')
        ax_ca.set_xlabel('Time(ms)')
        
        ax_cd = figure.add_subplot(gs[level,4:8])
        ax_cd.plot(t_down,cd[level],'ko--')
        ax_cd.set_title('{}º detail coefficient'.format(level))
        ax_cd.set_ylabel('Amplitude of coefficient (a.u.)')
        ax_cd.set_xlabel('Time(ms)')

    os.chdir('Figures')
    plt.tight_layout()
    plt.savefig(name+'.png')
    os.chdir('..')
    
    return None
    
if __name__=='__main__':

#Simulation input
    delta_time=0.5
    T2=100
    freq=0.1
    size_array=2048
    mean=0
    std_dev=0.1

#DWT input
    mother_wavelet='db'
    number_wavelet=8
    wavelet=mother_wavelet+'%d'%number_wavelet
    
    
    levels_dwt=3
    

    signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
    signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev,real=False)
    
    ca,cd = DWT_1D(signal_noise,wavelet,levels_dwt)

    name1='DWT1_fig3'
    First_figure(name1,size_array,delta_time,levels_dwt,ca,cd)
    First_figure_parts(name1,size_array,delta_time,levels_dwt,ca,cd)
    # name2='DWT2'
    # Second_figure(name2,size_array,delta_time,levels_dwt,ca,cd)
