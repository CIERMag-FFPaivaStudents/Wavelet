
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
Fix for sigma 5
"""

import numpy as np
import wavelet_1D
import signal_1D
import metrics_1D
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.optimize import curve_fit


def Line(x,a,b):
    y = b*x +a
    return y


def Sigma5(detail_coef_list,fix=False):
    """Median of the concatenation of the detail coefficients.
        
    Parameters
    ----------

        
    Return
    ------
        
    References
    ----------
    """
    median_factor = 0.675
    concatenated_detail_coef = np.hstack((detail_coef_list[1:]))
    concatenated_detail_coef = np.abs(concatenated_detail_coef)
    sigma5 = np.median(concatenated_detail_coef)
    
    if not fix:
        sigma5 /= median_factor
        

    return sigma5



if __name__ == '__main__':

    num_iterations=10000
    # num_iterations = 5
    # print('ble')
    
    delta_time=0.5
    T2=100
    freq=0.25 
    size_array=2048
    mean_set=0
    levels=5
    wavelet='db4'

    sigma_input_list=np.arange(0,0.6,0.01)
    
    length_input=len(sigma_input_list)



    sigma_output_mean_list=np.zeros((length_input,8))
    sigma_output_std_dev_list=np.zeros((length_input,8))

    for j in range(length_input):

        sigma_input=sigma_input_list[j]
        
        sigma_output_matrix=np.zeros((num_iterations,8))
        for i in range(num_iterations):

            signal_pure,time=signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq,False)
            
            signal_noise = signal_1D.Add_Noise1D(signal_pure,mean_set,sigma_input,False)


            approx_coef_list, detail_coef_list = wavelet_1D.DWT_1D(signal_noise,wavelet,levels)
            

            sigma_output_list = np.zeros(8)
            
            sigma_output_list[0] = sigma_input
            
            

            sigma_output_list[1]=Sigma5(detail_coef_list,fix=True)


            sigma_output_matrix[i]=sigma_output_list


        sigma_output_matrix = sigma_output_matrix.T

        sigma_output_mean=[sigma_output_list.mean() for sigma_output_list in sigma_output_matrix]
        sigma_output_mean = np.array(sigma_output_mean).T
        sigma_output_mean_list[j]=sigma_output_mean
    
        sigma_output_std_dev=[np.std(sigma_output_list) for sigma_output_list in sigma_output_matrix]
        sigma_output_std_dev = np.array(sigma_output_std_dev).T
        sigma_output_std_dev_list[j]=sigma_output_std_dev



    data_name = 'Sigma5_fix'
    parameters = '#Number of simulations={}, Mother Wavelet={}, Number of levels={},Array size={}'.format(num_iterations,wavelet,levels,size_array)
    metrics_name_list = ['sigma{}'.format(i+1) for i in range(7)]

    metrics_1D.Write_data(data_name,parameters,metrics_name_list,length_input,sigma_output_mean_list,sigma_output_std_dev_list)

    os.chdir('1D')
    os.chdir('Data')
    SIGMAS = np.loadtxt(data_name+'.txt').T
    
    m0,s0 = SIGMAS[0:2]
    m5,s5 = SIGMAS[2:4]
    
    plt.figure(figsize=(12,9))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')
    
    plt.errorbar(m0,m5,yerr=s5,xerr=None,marker='o')
    plt.xlabel(r'$\sigma_{input}$')
    plt.ylabel(r'$\sigma_5$')
    
    
    (a,b), mcov =curve_fit(Line,m5,m0)
    
    plt.plot(m0,Line(m0,a,b),'k-')
    
    print(num_iterations)
    print('\ta =',a)
    print('\tb =',b)
    os.chdir('..')






