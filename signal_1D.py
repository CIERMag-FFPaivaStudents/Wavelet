
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to simulate a MRI signal and its noise.
It also plots a figure of how it looks like with different parameters.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def Simulate_Signal1D(size_array,delta_time,T2,freq):
    time=np.arange(0, size_array*delta_time, delta_time)
    signal_pure=(np.exp(-time/T2))*np.cos(freq*time)
    return signal_pure, time

def Add_Noise1D(signal_pure,mean,std_dev):
    size_array=len(signal_pure)
    noise=np.random.normal(mean,std_dev,size_array)
    signal_noise=signal_pure+noise
    return signal_noise



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


    #Plotando a simulação de sinal
    
    len_xfig,len_yfig=3,3
    
    f, axes = plt.subplots(len_yfig, len_xfig,figsize=(8*len_xfig,4*len_yfig))
    
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')
    for i in range(3):
        for j in range(3):
    
            signal_pure,time=Simulate_Signal1D(size_array_set,delta_time_set,T2_list[i],freq_list[j])
            axes[i][j].plot(time,signal_pure, color='k')
            axes[i][j].set_ylabel('Amplitude do sinal (a.u.)')
            axes[i][j].set_xlabel('Tempo (ms)')
            axes[i][j].set_title('$T_2$={} ms, $\omega$={} KHz'.format(T2_list[i],freq_list[j]),fontsize=20)
    plt.tight_layout() 
    
    plt.savefig('freq.png')
    

    # # Plotando a simulação de ruído
    # len_xfig,len_yfig=3,3
    
    # f, axes = plt.subplots(len_yfig, len_xfig,figsize=(6*len_xfig,4*len_yfig))
    
    # sns.set()
    # sns.set_style('ticks')
    # sns.set_context('talk')
    # for i in range(3):
    #     for j in range(3):
    
    #         signal_pure,time=Simulate_Signal1D(size_array_set,delta_time_set,T2_set,freq_set)
    #         signal_noise = Add_Noise1D(signal_pure, mean_list[i],std_dev_list[j])
    #         axes[i][j].plot(time,signal_noise, color='k')
    #         axes[i][j].set_ylabel('Amplitude do sinal (a.u.)')
    #         axes[i][j].set_xlabel('Tempo (ms)')
    #         axes[i][j].set_title('$\mu$={} a.u. , $\sigma$={} a.u.'.format(mean_list[i],std_dev_list[j]))
    #         axes[i][j].set_ylim(-2.5,2.5)
    # plt.tight_layout()
    
    
    # plt.savefig('noise.svg')
    
plt.show() 


