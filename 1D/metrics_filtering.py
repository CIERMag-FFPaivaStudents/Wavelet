#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
Metrics of corresponding FFT graphs

"""

import numpy as np
import matplotlib.pyplot as plt
import signal_1D
import wavelet_1D
import sigmas
import metrics_1D
import filter_1D
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import shifted_filter
import seaborn as sns
from scipy import fft




def Metrics(signal1,signal2):
    
    mse = metrics_1D.MSE_Measure(signal1,signal2)
    prd = metrics_1D.PRD_Measure(signal1,signal2)
    snr = metrics_1D.SNR_Measure1D(signal1,signal2)

    
    return mse,prd,snr





if __name__ == '__main__':



    # os.chdir('Data')
    # os.chdir('..')

#Simulation input
    delta_time=0.5
    T2=100

    size_array=2048
    mean=0
    std_dev=0.1

    freq_list = [0,0.1,0.25,0.75]
    freq_title_list = ['000','010','025','075']
    
    num_iterations = 1000
    num_metrics = 6
    
    length_input = 25



#FID/FFT1
    for freq in freq_list:
        
        metrics_mean_list=np.zeros((length_input,num_metrics+1))
        metrics_std_dev_list=np.zeros((length_input,num_metrics+1))


        print(freq)


        metrics_matrix=np.zeros((num_iterations,num_metrics+1))
        
        for i in range(num_iterations):

            signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
            signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)
            

            M0_real= Metrics(np.real(signal_pure),np.real(signal_noise))
            M0_imag= Metrics(np.imag(signal_pure),np.imag(signal_noise))

            metrics_matrix[i] = np.array([0]+list(M0_real+M0_imag))
        
        metrics_matrix = metrics_matrix.T

        metrics_mean=[metrics_list.mean() for metrics_list in metrics_matrix]
        metrics_mean = np.array(metrics_mean).T
        metrics_mean_list[0]=metrics_mean

        metrics_std_dev=[np.std(metrics_list) for metrics_list in metrics_matrix]
        metrics_std_dev = np.array(metrics_std_dev).T
        metrics_std_dev_list[0]=metrics_std_dev

        


        for num_params in range(1,25):
            
            
            
            print('\t%.d'%num_params)
            os.chdir('Data')
            params_name = 'Parameters'
            params = open(params_name+'.txt').readlines()[num_params]
            os.chdir('..')
            params = params.split('\t')
            
            noise_estimator, mode, wavelet, levels_dwt = params[1:]
            noise_estimator,levels_dwt = int(noise_estimator),int(levels_dwt)



            
            metrics_matrix=np.zeros((num_iterations,num_metrics+1))
            
            for i in range(num_iterations):
                signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
                signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)

                signal_smooth = filter_1D.Wavelet_filter(signal_noise,wavelet,levels_dwt,mode,'SURE')


                M_real= Metrics(np.real(signal_pure),np.real(signal_smooth))
                M_imag= Metrics(np.imag(signal_pure),np.imag(signal_smooth))

            
            
                metrics_matrix[i] = np.array([num_params]+list(M_real+M_imag))
                
            metrics_matrix = metrics_matrix.T
    
            metrics_mean=[metrics_list.mean() for metrics_list in metrics_matrix]
            metrics_mean = np.array(metrics_mean).T
            metrics_mean_list[num_params]=metrics_mean
    
            metrics_std_dev=[np.std(metrics_list) for metrics_list in metrics_matrix]
            metrics_std_dev = np.array(metrics_std_dev).T
            metrics_std_dev_list[num_params]=metrics_std_dev
    
        data_name = 'FID_FFT1_'+freq_title_list[freq_list.index(freq)]
        metrics_name_list =['MSE_real','SNR_real','PRD_real','MSE_imag','SNR_imag','PRD_imag']
        parameters  = '#num_iterations={}'.format(num_iterations)
        
        # metrics_1D.Write_data(data_name,parameters,metrics_name_list,length_input,metrics_mean_list,metrics_std_dev_list)




#FID/FFT2

    length_input=24
    metrics_mean_list=np.zeros((length_input,num_metrics+1))
    metrics_std_dev_list=np.zeros((length_input,num_metrics+1))

        
    for num_params in range(25,49):
        print(num_params)
        os.chdir('Data')
        params_name = 'Parameters2'
        params = open(params_name+'.txt').readlines()[num_params-24]
        os.chdir('..')
        
        params = params.split('\t')

        freq,mode, wavelet, levels_dwt = params[1:]
        freq,levels_dwt = float(freq),int(levels_dwt)
        
        metrics_matrix=np.zeros((num_iterations,num_metrics+1))
        
        for i in range(num_iterations):


            signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
        
            signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)
    
            signal_smooth = filter_1D.Wavelet_filter(signal_noise,wavelet,levels_dwt,mode,'SURE')


            M_real= Metrics(np.real(signal_pure),np.real(signal_smooth))
            M_imag= Metrics(np.imag(signal_pure),np.imag(signal_smooth))
    
            
            
            metrics_matrix[i] = np.array([num_params]+list(M_real+M_imag))
                
        metrics_matrix = metrics_matrix.T

        metrics_mean=[metrics_list.mean() for metrics_list in metrics_matrix]
        metrics_mean = np.array(metrics_mean).T
        metrics_mean_list[num_params-25]=metrics_mean

        metrics_std_dev=[np.std(metrics_list) for metrics_list in metrics_matrix]
        metrics_std_dev = np.array(metrics_std_dev).T
        metrics_std_dev_list[num_params-25]=metrics_std_dev
    
    data_name = 'FID_FFT2'
    metrics_name_list =['MSE_real','SNR_real','PRD_real','MSE_imag','SNR_imag','PRD_imag']
    parameters  = '#num_iterations={}'.format(num_iterations)
    
    # metrics_1D.Write_data(data_name,parameters,metrics_name_list,length_input,metrics_mean_list,metrics_std_dev_list)
    os.chdir('1D')
    
    
    
#FID/FFT3
    alpha_list=10*np.arange(0,11,1)
    
    for alpha in alpha_list:
        
        metrics_mean_list=np.zeros((length_input,num_metrics+1))
        metrics_std_dev_list=np.zeros((length_input,num_metrics+1))


        print(alpha)

        


        for num_params in range(25,49):
            
            
            
            os.chdir('Data') 
            params_name = 'Parameters2'
            params = open(params_name+'.txt').readlines()[num_params-24]
            os.chdir('..')
            
            params = params.split('\t')
    
    
            freq,mode, wavelet, levels_dwt = params[1:]
            freq,levels_dwt = float(freq),int(levels_dwt)


            metrics_matrix=np.zeros((num_iterations,num_metrics+1))
            
            for i in range(num_iterations):
                signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
            
                signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev, False)
                
                signal_smooth = filter_1D.Wavelet_filter(signal_noise,wavelet,levels_dwt,mode,'SURE',alpha=alpha)
            

                M_real= Metrics(np.real(signal_pure),np.real(signal_smooth))
                M_imag= Metrics(np.imag(signal_pure),np.imag(signal_smooth))

            
            
                metrics_matrix[i] = np.array([num_params]+list(M_real+M_imag))
                
            metrics_matrix = metrics_matrix.T
    
            metrics_mean=[metrics_list.mean() for metrics_list in metrics_matrix]
            metrics_mean = np.array(metrics_mean).T
            metrics_mean_list[num_params-25]=metrics_mean
    
            metrics_std_dev=[np.std(metrics_list) for metrics_list in metrics_matrix]
            metrics_std_dev = np.array(metrics_std_dev).T
            metrics_std_dev_list[num_params-25]=metrics_std_dev
    
        data_name = 'FID_FFT3_{}'.format(alpha)
        metrics_name_list =['MSE_real','SNR_real','PRD_real','MSE_imag','SNR_imag','PRD_imag']
        parameters  = '#num_iterations={}'.format(num_iterations)
        
        metrics_1D.Write_data(data_name,parameters,metrics_name_list,length_input,metrics_mean_list,metrics_std_dev_list)
        os.chdir('1D')
        
        
        
        
#FID/FFT4

    length_input=24
    metrics_mean_list=np.zeros((length_input,num_metrics+1))
    metrics_std_dev_list=np.zeros((length_input,num_metrics+1))

        
    for num_params in range(25,49):
        print(num_params)
        os.chdir('Data')
        params_name = 'Parameters2'
        params = open(params_name+'.txt').readlines()[num_params-24]
        os.chdir('..')
        
        params = params.split('\t')

        freq,mode, wavelet, levels_dwt = params[1:]
        freq,levels_dwt = float(freq),int(levels_dwt)
        
        metrics_matrix=np.zeros((num_iterations,num_metrics+1))
        
        for i in range(num_iterations):


            signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
        
            signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,std_dev,False)
        
            shifted_signal,freq_peak = shifted_filter.shift_freq(signal_noise, time, delta_time)
        
            signal_smooth = filter_1D.Wavelet_filter(shifted_signal,wavelet,levels_dwt,mode,'SURE')
        
            signal_smooth_reshift = signal_smooth*np.exp(1j*2*np.pi*freq_peak*time)


            M_real= Metrics(np.real(signal_pure),np.real(signal_smooth_reshift))
            M_imag= Metrics(np.imag(signal_pure),np.imag(signal_smooth_reshift))
    
            
            
            metrics_matrix[i] = np.array([num_params]+list(M_real+M_imag))
                
        metrics_matrix = metrics_matrix.T

        metrics_mean=[metrics_list.mean() for metrics_list in metrics_matrix]
        metrics_mean = np.array(metrics_mean).T
        metrics_mean_list[num_params-25]=metrics_mean

        metrics_std_dev=[np.std(metrics_list) for metrics_list in metrics_matrix]
        metrics_std_dev = np.array(metrics_std_dev).T
        metrics_std_dev_list[num_params-25]=metrics_std_dev
    
    data_name = 'FID_FFT4'
    metrics_name_list =['MSE_real','SNR_real','PRD_real','MSE_imag','SNR_imag','PRD_imag']
    parameters  = '#num_iterations={}'.format(num_iterations)
    
    metrics_1D.Write_data(data_name,parameters,metrics_name_list,length_input,metrics_mean_list,metrics_std_dev_list)
    os.chdir('1D')