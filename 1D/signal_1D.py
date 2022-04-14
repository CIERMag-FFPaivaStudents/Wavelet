
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to simulate a MRI signal and its noise.
It also plots a figure of how it looks like with different parameters.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def Simulate_Signal1D(size_array,delta_time,T2,freq):
    """Simulates the free induction decay (FID) signal of the magnetic ressonance phenomenon as described in Brown et al and Mazzola.
    
    Parameters
    ----------
    size_array: int
        Desired length of the simulated 1D signal.
        
    delta_time: float
        Interval between consecutive points of time.
        
    T2: float
        Transversal relaxation time of tissue.
        
    freq: float
        Precession frequency of hidrogen atoms in the tissue.
        
    Return
    ------
    signal_pure: array
        Noiseless simulated free induction decay 1D signal.
        
    time: array
        Time array of the corresponding points in the signal.
        
    References
    ----------
        BROWN, Robert W.; CHENG, Yu Chung N.; HAACKE, E. Mark; THOMPSON,
        Michael R.; VENKATESAN, Ramesh. Magnetic Resonance Imaging: Physical
        Principles and Sequence Design: Second Edition. [s.l: s.n.]. v. 9780471720850
        DOI: 10.1002/9781118633953
        
        MAZZOLA, Alessandro A. Ressonância magnética: princípios de formação da
        imagem e aplicações em imagem funcional. Revista Brasileira de Física Médica, [S.
        l.], v. 3, n. 1, p. 117–129, 2009. DOI: https://doi.org/10.29384/rbfm.2009.v3.n1.p117-
        129
    """
    time=np.arange(0, size_array*delta_time, delta_time)
    signal_pure=(np.exp(-time/T2))*np.cos(freq*time)
    return signal_pure, time

def Add_Noise1D(signal_pure,mean,std_dev):
    """Adds gaussian noise to a 1D free induction decay signal as described in Aja-Fernández et al.
    Parameters
    ----------
    signal_pure: array 
        Noiseless free induction decay 1D signal.
        
    mean: float
        Center of the Gaussian noise distribution which will be added to the signal.
        
    std_dev: float
        Width of the Gaussian noise distribution which will be added to the signal.
        Must be non-negative
        
    Return
    ------
    signal_noise: array
        Noisy simulated free induction decay 1D signal.

    References
    ----------
        AJA-FERNÁNDEZ, Santiago; VEGAS-SÁNCHEZ-FERRERO, Gonzalo.
        Statistical Analysis of Noise in MRI. Cham: Springer International Publishing, 2016.
        DOI: 10.1007/978-3-319-39934-8. Available at:
        http://link.springer.com/10.1007/978-3-319-39934-8.
    """
    size_array=len(signal_pure)
    noise=np.random.normal(mean,std_dev,size_array)
    signal_noise=signal_pure+noise
    return signal_noise



def Signal_figure(name,size_array_set,delta_time_set,T2_list,freq_list):
    """Plots a figure designed to show the influences of the signal parameters and creates a .png image of it.

    Parameters
    ----------

    name: string
        Desired name of the image.
    
    size_array_set: int
        Set desired length of the simulated 1D signal in all the subplots.
    
    delta_time_set: float
        Set interval between consecutive points of time in all the subplots.
    
    T2_list: array
        Array with transversal relaxation times of several tissues.
    
    freq_list: array
        Array with precession frequencies of hidrogen atoms in several tissues.
    
    Return
    ------

    References
    ----------
    """
    len_xfig,len_yfig=3,3
    
    f, axes = plt.subplots(len_yfig, len_xfig,figsize=(8*len_xfig,4*len_yfig))
    
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')
    for i in range(3):
        for j in range(3):
            signal_pure,time=Simulate_Signal1D(size_array_set,delta_time_set,T2_list[i],freq_list[j])
            axes[i][j].plot(time,signal_pure, color='k')
            axes[i][j].set_ylabel('Signal Amplitude (a.u.)')
            axes[i][j].set_xlabel('Time (ms)')
            axes[i][j].set_title('$T_2$={} ms, $\omega$={} KHz'.format(T2_list[i],freq_list[j]),fontsize=20)
    plt.tight_layout() 
    os.chdir('Figures')
    plt.savefig(name+'.png')
    os.chdir('..')

    return None


def Noise_figure(name,size_array_set,delta_time_set,T2_set,freq_set,mean_list,std_dev_list):
    """Plots a figure designed to show the influences of the noise parameters and creates a .png image of it.

    Parameters
    ----------

    name: string
        Desired name of the image.
    
    size_array_set: int
        Set desired length of the simulated 1D signal in all the subplots.
    
    delta_time_set: float
        Set interval between consecutive points of time in all the subplots.
    
    T2_set: float
        Set transversal relaxation times of a tissue in all the subplots.
    
    freq_set: float
        Set precession frequency of hidrogen atoms all the subplots.
    
    mean_list: array
        Array with the centers of the Gaussian noise distributions which will be added to several signals.
    
    std_dev_list: array
        Array with the widths of the Gaussian noise distributions which will be added to several signals.
        Each object of this array must be non-negative

    Return
    ------

    References
    ----------
    """
    len_xfig,len_yfig=3,3
    
    f, axes = plt.subplots(len_yfig, len_xfig,figsize=(6*len_xfig,4*len_yfig))
    
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')
    for i in range(3):
        for j in range(3):
            signal_pure,time=Simulate_Signal1D(size_array_set,delta_time_set,T2_set,freq_set)
            signal_noise = Add_Noise1D(signal_pure, mean_list[i],std_dev_list[j])
            axes[i][j].plot(time,signal_noise, color='k')
            axes[i][j].set_ylabel('Signal Amplitude (a.u.)')
            axes[i][j].set_xlabel('Time (ms)')
            axes[i][j].set_title('$\mu$={} a.u. , $\sigma$={} a.u.'.format(mean_list[i],std_dev_list[j]))
            axes[i][j].set_ylim(-2.5,2.5)
    plt.tight_layout()
    os.chdir('Figures')
    plt.savefig(name+'.png')
    os.chdir('..')
    return None



if __name__ == '__main__':
    delta_time_set=0.5
    T2_set=100
    freq_set=0.25 
    size_array_set=2048
    mean_set=0
    std_dev_set=0.01


    T2_list=[40,T2_set,2000]
    freq_list =[0.125, 0.25, 0.5]

    
    
    mean_list = [-1,mean_set,1]
    std_dev_list=[std_dev_set,0.1,0.5]


    name_signal='Signal figure'
    Signal_figure(name_signal,size_array_set,delta_time_set,T2_list,freq_list)

    name_noise='Noise figure'
    Noise_figure(name_noise,size_array_set,delta_time_set,T2_set,freq_set,mean_list,std_dev_list)


