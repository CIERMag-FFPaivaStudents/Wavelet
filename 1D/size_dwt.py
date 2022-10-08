
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to analyse the size of dwt coefficients.
"""
import signal_1D
import wavelet_1D
import sigmas
import metrics_1D
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import fft
import seaborn as sns
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



#simulating signal
    signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq, False)
    signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)



#DWT on simulated signal
    coefficients = pywt.wavedec(signal_noise, wavelet, level=levels_dwt)
    signal_test = pywt.waverec(coefficients, wavelet)

    approx_coeff= coefficients[0]
    detail_coeffs = coefficients[1:] 

#printing the length of each detail coefficient
    for i in range(len(detail_coeffs)-1,-1,-1):
        print(len(detail_coeffs[i]))


    print('size of signal array\t',size_array)
    print('wavelet used\t',wavelet)
    print('levels of DWT\t',levels_dwt)
    print('length of first detal coefficient\t',len(detail_coeffs[-1]))
    


#characteristic of mother wavelet
    w = pywt.Wavelet(wavelet)
    print('\nwavelet filter length\t',w.dec_len)

    max_level = pywt.dwt_max_level(size_array, w.dec_len)
    print('max level of mother wavelet\t',max_level)


#showing the last detail coefficient
    plt.figure()
    coef_detail=detail_coeffs[1]
    plt.plot(np.real(coef_detail),'ro-',ms=4)
    plt.ylabel('Amplitude (a.u.)')
    plt.xlabel('Position in array')
    print('length of last detal coefficient\t',len(coef_detail))
    plt.show()



    
#     mother_list = ['db','sym','coif']

# #max and min support for each mother wavelet as seen on pywt.wavelist(mother wavelet),
# #the order is the same as in mother_list

#     support_lim = [1,38,2,20,1,17]
    
    
#     support_list = []
#     dec_len_list = []
#     max_level_list = []
    
#     plt.clf()
    
#     fig=plt.figure(figsize=(15,7))

#     sns.set()
#     sns.set_context('talk')
#     sns.set_style('white')
    
    
    
#     ax1 = fig.add_subplot(1,2,1)
#     ax2 = fig.add_subplot(1,2,2)
    
#     plot_symbol = ['r^','g<','bo']
    
#     for m in range(len(mother_list)):

#         support_min,support_max = support_lim[2*m:2*(m+1)]
#         support  = np.arange(support_min,support_max+1)
#         support_list.append(support)
    
        
#         dec_len = np.zeros(support_max+1-support_min)
#         max_level = np.zeros(support_max+1-support_min)
        
#         for s in range(support_min,support_max+1):
#             wavelet = mother_list[m]+'%.d'%(s)
#             wavelet = pywt.Wavelet(wavelet)
            
#             dec_len[s-support_min] = wavelet.dec_len
#             max_level[s-support_min] = pywt.dwt_max_level(size_array, wavelet.dec_len)
        
#         dec_len_list.append(dec_len)
#         max_level_list.append(max_level)
        
#         ax1.plot(support, dec_len,plot_symbol[m],label=mother_list[m])
#         ax1.set_xlabel('Support')
#         ax1.set_ylabel('Filter length')

#         ax2.plot(support, max_level,plot_symbol[m],label=mother_list[m])
#         ax2.set_xlabel('Support')
#         ax2.set_ylabel('Max level of DWT')

#         ax2.plot([5,10,17,8,16,20],[6,5,4,7,6,5],'ko',ms=10)
        
        
#     plt.legend()
#     plt.tight_layout()
    
#     os.chdir('Graphs')
#     # plt.savefig('Wavelet_Support.png')
        

