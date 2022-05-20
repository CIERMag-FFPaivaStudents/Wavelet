
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to filter a noisy MRI signal.
"""
import signal_1D
import wavelet_1D
import sigmas
import metrics_1D
import numpy as np
import matplotlib.pyplot as plt
import pywt


def Std_dev_estimator(signal_noise,detail_coef_list,num=5):
    """
    
    Parameters
    ----------


    Return
    ------

    References
    ----------
    """

    if num==2:
        cut_point = 30
        sigma=sigmas.Sigma2(signal_noise,cut_point)

    elif num==5:
        sigma = sigmas.Sigma5(detail_coef_list)

    return sigma


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
        approx_coef_hat = pywt.idwt(approx_coef_hat_list[level], detail_coef_hat_list[level], wavelet)
        
        if level<levels:
            ld , la = len(detail_coef_hat_list[level+1]),len(approx_coef_hat)
            if la>ld:
                approx_coef_hat_list[level+1] = approx_coef_hat[:-1]
            else:
                approx_coef_hat_list[level+1] = approx_coef_hat

    signal_hat = approx_coef_hat_list[-1]
    return signal_hat



def Thresholding(data,T,mode):
    """
    
    Parameters
    ----------


    Return
    ------

    References
    ----------
    """
    
    output = data.copy()
    if mode=='hard':
        output[np.abs(data)<T]=0
        return output
    elif mode == 'soft':
        output[data>-T]-=T
        output[data<T]+=T
        output[np.abs(data)<=T] = 0
        return output
    else:
        print('Erro')
        return None


def Adaptive_smoothing(signal,detail_coef_list,mode,num=5,alg='SURE'):
    """
    
    Parameters
    ----------


    Return
    ------

    References
    ----------
    """
    
    sigma = Std_dev_estimator(signal,detail_coef_list,num)
    smooth_detail_coef_list = len(detail_coef_list)*[None]
    for i in range(len(detail_coef_list)):
        size_coef = len(detail_coef_list[i])
        
        if alg == 'SURE':
            T=sigma*np.sqrt(2*np.log(size_coef))
        elif alg == 'Bayes':
            T = sigma**2 / np.std(signal)

        smooth_detail_coef= Thresholding(detail_coef_list[i],T,mode)
        smooth_detail_coef_list[i] = smooth_detail_coef
    return smooth_detail_coef_list







if __name__ == "__main__":
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
    
    ca,cd = wavelet_1D.DWT_1D(signal_noise,wavelet,levels_dwt)
    SNR0=metrics_1D.SNR_Measure1D(signal_pure,signal_noise)
    # print('SNR0',SNR0)


    signal_test = IDWT_1D(ca[-1],cd,wavelet,levels_dwt)
    SNR_test=metrics_1D.SNR_Measure1D(signal_pure,signal_test)
    # plt.plot(ca_hat[1],'bo-')
    # plt.plot(ca[2],'ro')
    # plt.xlim(500,520)
    
    
    # plt.plot(time,signal_noise,'b-')
    # plt.plot(time,signal_test,'ro')
    # plt.xlim(2040,2050)
    # plt.xlim(-5,5)

    mode = 'soft'

    cd_smooth = Adaptive_smoothing(signal_pure,cd,mode,alg='Bayes')
    signal_smooth = IDWT_1D(ca[-1],cd_smooth,wavelet,levels_dwt)

    SNR1=metrics_1D.SNR_Measure1D(signal_pure,signal_smooth)
    # plt.plot(signal_noise,signal_test,'bo')
    # plt.plot(signal_noise,signal_smooth,'r-')

    # plt.xlim(-0.1,0.1)
    fig=plt.figure()
    ax1=fig.add_subplot(2,1,1)
    ax1.plot(time,signal_noise)

    ax1=fig.add_subplot(2,1,2)
    ax1.plot(time,signal_smooth)
    
    print(std_dev,SNR0,SNR1,(SNR1-SNR0)/SNR0)
