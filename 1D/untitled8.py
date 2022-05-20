
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
Arena
"""
import signal_1D
import wavelet_1D
import sigmas
import metrics_1D
import numpy as np
import matplotlib.pyplot as plt
import pywt
import timeit


def IDWT_1D(last_approx_coef,detail_coef_list,wavelet,levels):
    """
    
    Parameters
    ----------


    Return
    ------

    References
    ----------
    """
    
    detail_coef_hat_list = detail_coef_list.copy()
    detail_coef_hat_list.reverse()
    
    approx_coef_hat_list= (levels+1)*[None]
    
    approx_coef_hat_list[0] = last_approx_coef

    for level in range(levels+1):
        print(len(detail_coef_hat_list[level]),len(approx_coef_hat_list[level]))
        approx_coef_hat = pywt.idwt(approx_coef_hat_list[level], detail_coef_hat_list[level], wavelet)
        # approx_coef_hat_list[level+1] = approx_coef_hat[:-1]
        
        if level<levels:
            ld , la = len(detail_coef_hat_list[level+1]),len(approx_coef_hat)
            if la>ld:
                approx_coef_hat_list[level+1] = approx_coef_hat[:-1]
        else:
            signal_hat = pywt.idwt(approx_coef_hat_list[level], detail_coef_hat_list[level], wavelet)

    return None# signal_hat






#Simulation input
delta_time=0.5
T2=100
freq=0.25
size_array=2048
mean=0
std_dev=0.05

#DWT input
mother_wavelet='db'
number_wavelet=4
wavelet=mother_wavelet+'%d'%number_wavelet


levels_dwt = 6


signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq)
signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev)



L=10000
start1 = timeit.default_timer()

for beta in range(L):
    ca,cd = wavelet_1D.DWT_1D(signal_noise,wavelet,levels_dwt)
    levels = levels_dwt
    detail_coef_list = cd
    last_approx_coef  = ca[-1]
    
    detail_coef_hat_list = detail_coef_list.copy()
    detail_coef_hat_list.reverse()
    
    approx_coef_hat_list= (levels+1)*[None]
    
    approx_coef_hat_list[0] = last_approx_coef
    
    for level in range(levels+1):
        # print(level)
        # print(len(detail_coef_hat_list[level]),len(approx_coef_hat_list[level]))
        approx_coef_hat = pywt.idwt(approx_coef_hat_list[level], detail_coef_hat_list[level], wavelet)
        # approx_coef_hat_list[level+1] = approx_coef_hat[:-1]
        
        if level<levels:
            ld , la = len(detail_coef_hat_list[level+1]),len(approx_coef_hat)
            if la>ld:
                approx_coef_hat_list[level+1] = approx_coef_hat[:-1]
            else:
                approx_coef_hat_list[level+1] = approx_coef_hat
        else:
            signal_hat = pywt.idwt(approx_coef_hat_list[level], detail_coef_hat_list[level], wavelet)

stop1 = timeit.default_timer()
time1 =stop1 - start1


start2 = timeit.default_timer()

for beta in range(L):
    coeffs = pywt.wavedec(signal_noise, wavelet, level=levels_dwt)
    signal_hat_pc = pywt.waverec(coeffs, wavelet)

stop2 = timeit.default_timer()
time2 =stop2 - start2


print(time1/time2)