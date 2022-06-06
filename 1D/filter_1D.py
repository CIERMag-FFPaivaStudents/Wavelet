
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
from scipy import fft

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
    
    # print('sigma',sigma)
    return sigma


def Wavelet_filter(signal_noise,wavelet,levels_dwt,mode,alg):

    coefficients = pywt.wavedec(signal_noise, wavelet, level=levels_dwt)
    ca = coefficients[0]
    cd = coefficients[1:]

    cd_smooth = Adaptive_smoothing(signal_noise,cd,mode,alg=alg)
    
    coefficients_smooth  = [ca]+cd_smooth
    signal_smooth = pywt.waverec(coefficients_smooth,wavelet)
    
    return signal_smooth

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
            # T=sigma*np.sqrt(2*np.log(2048))
            
            # print('\tsigma',sigma)
            # print(T,detail_coef_list[i].min(),detail_coef_list[i].max(),size_coef)


        elif alg == 'Bayes':
            T = sigma**2 / np.std(signal)
            # print(T,detail_coef_list[i].min(),detail_coef_list[i].max())


        smooth_detail_coef= Thresholding(detail_coef_list[i],T,mode)
        smooth_detail_coef_list[i] = smooth_detail_coef
    return smooth_detail_coef_list



if __name__ == "__main__":
    #Simulation parameters
    delta_time=0.5
    T2=100
    freq=0.25
    size_array=2048
    mean=0
    std_dev=0.1

    #DWT parameters
    mother_wavelet='db'
    number_wavelet=4
    wavelet=mother_wavelet+'%d'%number_wavelet
    
    
    levels_dwt = 4
   
    mode = 'soft'
    alg = 'SURE'

    signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq, False)
    signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)

    plt.figure()
    plt.plot(time, np.real(signal_noise))
    plt.plot(time, np.imag(signal_noise))
    plt.show()


    SNR0=metrics_1D.SNR_Measure1D(signal_pure,signal_noise)


    coefficients = pywt.wavedec(signal_noise, wavelet, level=levels_dwt)


    signal_test = pywt.waverec(coefficients, wavelet)
    SNR_test=metrics_1D.SNR_Measure1D(signal_pure,signal_test)

    ca = coefficients[0]
    cd = coefficients[1:] 


    cd_smooth = Adaptive_smoothing(signal_noise,cd,mode,alg=alg)
    
    coefficients_smooth  = [ca]+cd_smooth
    signal_smooth = pywt.waverec(coefficients_smooth,wavelet)

    # signal_smooth_func = Wavelet_filter(signal_noise, wavelet, levels_dwt, mode, alg)

    SNR1=metrics_1D.SNR_Measure1D(signal_pure,signal_smooth)


    fig=plt.figure(figsize=(20,20))
    ax1=fig.add_subplot(2,1,1)
    ax1.plot(time,np.real(signal_noise))
    ax1.set_title('Noisy signal',fontsize=20)

    ax1=fig.add_subplot(2,1,2)
    ax1.plot(time,np.real(signal_smooth))
    ax1.set_title('Filtered signal',fontsize=20)

    print('To check if the SNR increased.')
    print(std_dev,SNR0,SNR1)
    
    plt.figure()
    size_array = len(signal_noise)
    yf=fft.fft(signal_noise)
    xf=fft.fftfreq(size_array,delta_time)
    xf=fft.fftshift(xf)
    yplot=fft.fftshift(yf)
    plt.plot(xf, yplot.real)
    plt.plot(xf, yplot.imag)
    plt.plot(xf, np.abs(yplot))

    plt.figure()
    size_array = len(signal_smooth)
    yf=fft.fft(signal_smooth)
    xf=fft.fftfreq(size_array,delta_time)
    xf=fft.fftshift(xf)
    yplot=fft.fftshift(yf)
    plt.plot(xf, yplot.real)
    plt.plot(xf, yplot.imag)
    plt.plot(xf, np.abs(yplot))
    plt.show()
