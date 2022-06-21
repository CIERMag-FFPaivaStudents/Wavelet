
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to analyse the size of an array after a dwt
"""
import signal_1D
import wavelet_1D
import sigmas
import metrics_1D
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import fft
import os



if __name__ == "__main__":
    #Simulation parameters
    delta_time=0.5
    T2=100
    freq=0.25
    size_array=2048
    mean=0
    std_dev=0.1

    #DWT parameters
    mother_wavelet='sym'
    number_wavelet=20
    wavelet=mother_wavelet+'%d'%number_wavelet
    
    
    
    
    levels_dwt = 15

    mode = 'soft'
    alg = 'SURE'

    signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq, False)
    signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)





    

    SNR0=metrics_1D.SNR_Measure1D(signal_pure,signal_noise)


    coefficients = pywt.wavedec(signal_noise, wavelet, level=levels_dwt)


    signal_test = pywt.waverec(coefficients, wavelet)
    SNR_test=metrics_1D.SNR_Measure1D(signal_pure,signal_test)

    ca = coefficients[0]
    cd = coefficients[1:] 


    # l = size_array/2**levels_dwt
    for i in range(len(cd)):
        print(len(cd[-i-1]))
        
    print('size_array\t',size_array)
    print('wavelet\t',wavelet)
    print('levels\t',levels_dwt)
    print('len cd\t',len(cd[-1]))
    
    
    


    w = pywt.Wavelet(wavelet)
    
    print('filter length\t',w.dec_len)
    # 
    max_level = pywt.dwt_max_level(size_array, w.dec_len)
    print('max level\t',max_level)


    
    plt.figure()
    coef_detail=cd[1]
    plt.plot(np.real(coef_detail),'ro-',ms=4)
    print(len(coef_detail))
    plt.show()



    
    mother_list = ['db','sym','coif']

#max and min support for each mother wavelet as seen on pywt.wavelist(mother wavelet), the order is the same as in mother_list
    support_lim = [1,38,2,20,1,17]
    
    
    support_list = []
    dec_len_list = []
    max_level_list = []
    
    plt.clf()
    
    fig=plt.figure(figsize=(22,8))
    
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    for m in range(len(mother_list)):

        support_min,support_max = support_lim[2*m:2*(m+1)]
        support  = np.arange(support_min,support_max+1)
        support_list.append(support)
    
        
        dec_len = np.zeros(support_max+1-support_min)
        max_level = np.zeros(support_max+1-support_min)
        
        for s in range(support_min,support_max+1):
            wavelet = mother_list[m]+'%.d'%(s)
            # print(wavelet)
            wavelet = pywt.Wavelet(wavelet)
            
            dec_len[s-support_min] = wavelet.dec_len
            max_level[s-support_min] = pywt.dwt_max_level(size_array, wavelet.dec_len)
        
        dec_len_list.append(dec_len)
        max_level_list.append(max_level)
        
        ax1.plot(support, dec_len,'o',label=mother_list[m])
        ax1.set_xlabel('Support',fontsize=15)
        ax1.set_ylabel('Filter length',fontsize=15)
        

        ax2.plot(support, max_level,'o',label=mother_list[m])
        ax2.set_xlabel('Support',fontsize=15)
        ax2.set_ylabel('Max level of dwt',fontsize=15)
        
    plt.legend(fontsize=13)
    plt.tight_layout()
    
    os.chdir('Graphs')
    plt.savefig('Wavelet_Support.png')
        

