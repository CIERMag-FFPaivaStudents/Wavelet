
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


    l = size_array/2**levels_dwt
    for i in range(len(cd)):
        print(len(cd[-i-1]))
        
    print('size_array\t',size_array)
    print('wavelet\t',wavelet)
    print('levels\t',levels_dwt)
    print('len cd\t',len(cd[-1]))
    print('len_est\t',l)
    
    
    
    plt.figure()
    coef_detail=cd[1]
    plt.plot(np.real(coef_detail),'ro-',ms=4)
    print(len(coef_detail))
    
    plt.show()

