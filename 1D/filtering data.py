
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is plot graphs of filtering.
"""
import signal_1D
import wavelet_1D
import sigmas
import metrics_1D
import filter_1D
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os






if __name__ == "__main__":
#Simulation input
    delta_time=0.5
    T2=100
    freq=0.25
    size_array=2048
    mean=0


    num_iterations = 10000
#DWT input

    # num_params=22
    # print(os.getcwd())
    for num_params in range(1,25):
        print(num_params)
        # print(os.getcwd())
        os.chdir('Data')
        params = open('Parameters.txt').readlines()[num_params]

        params = params.split('\t')
        noise_estimator, mode, wavelet, levels_dwt = params[1:]
        noise_estimator,levels_dwt = int(noise_estimator),int(levels_dwt)
        # print(noise_estimator, mode, wavelet, levels_dwt)
    

        num_metrics = 6
        
        sigma_input_list=np.arange(0,1,0.05)
    
        length_input=len(sigma_input_list)
    
    
        metrics_mean_list=np.zeros((length_input,num_metrics+1))
        metrics_std_dev_list=np.zeros((length_input,num_metrics+1))
        
        for j in range(length_input):
            sigma_input=sigma_input_list[j]
            
            metrics_matrix=np.zeros((num_iterations,num_metrics+1))
            
            for i in range(num_iterations):
    
                signal_pure,time = signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,real=False)
                signal_noise = signal_1D.Add_Noise1D(signal_pure,mean,sigma_input, False)

                
                MSE0=metrics_1D.MSE_Measure(signal_pure, signal_noise)
                
                SNR0=metrics_1D.SNR_Measure1D(signal_pure,signal_noise)
    
                PRD0=metrics_1D.PRD_Measure(signal_pure,signal_noise)
    
                
                
                alg = 'SURE'
                signal_smooth = filter_1D.Wavelet_filter(signal_noise,wavelet,levels_dwt,mode,alg)
                
                MSE1=metrics_1D.MSE_Measure(signal_pure,signal_smooth)
                
                SNR1=metrics_1D.SNR_Measure1D(signal_pure,signal_smooth)
    
                PRD1=metrics_1D.PRD_Measure(signal_pure,signal_smooth)
                
                metrics_list = np.zeros(num_metrics+1)
                
                metrics_list[0] = sigma_input
                
                
                metrics_list[1] = MSE0
                metrics_list[2] = MSE1
                metrics_list[3] = SNR0
                metrics_list[4] = SNR1
                metrics_list[5] = PRD0
                metrics_list[6] = PRD1
    
                metrics_matrix[i]=metrics_list
    
            metrics_matrix = metrics_matrix.T
    
            metrics_mean=[metrics_list.mean() for metrics_list in metrics_matrix]
            metrics_mean = np.array(metrics_mean).T
            metrics_mean_list[j]=metrics_mean
    
            metrics_std_dev=[np.std(metrics_list) for metrics_list in metrics_matrix]
            metrics_std_dev = np.array(metrics_std_dev).T
            metrics_std_dev_list[j]=metrics_std_dev
    
    
        data_name = 'Metrics_filter1D {}'.format(num_params)
        metrics_name_list =['MSE0','MSE1','SNR0','SNR1','PRD0','PRD1']
        parameters  = ''
        

        os.chdir('..')
        metrics_1D.Write_data(data_name,parameters,metrics_name_list,length_input,metrics_mean_list,metrics_std_dev_list,folder='/Metrics_filter')

        os.chdir('..')

        os.chdir('1D')
