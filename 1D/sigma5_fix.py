
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to fix the sigma 5 estimator (see sigmas.py) for a complex signal input.
"""

import numpy as np
import wavelet_1D
import signal_1D
import metrics_1D
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.optimize import curve_fit
import pywt

def Line(x,a,b):
    y = a*x +b
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
    # else:
        # a=1.1846438408877022
        # b=0.006499950443242419
        # sigma5 = (sigma5-b)/a
        

    return sigma5



if __name__ == '__main__':

    num_iterations=10000

    
    delta_time=0.5
    T2=100
    freq=0.50 
    size_array=2048
    mean_set=0
    levels=5
    wavelet='db4'

    sigma_input_list=np.arange(0,0.6,0.01)
    
    length_input=len(sigma_input_list)

    freq_list = [0,0.1,0.25,0.75]

    num_var=4
    sigma_output_mean_list=np.zeros((length_input,1+num_var))
    sigma_output_std_dev_list=np.zeros((length_input,1+num_var))

    for j in range(length_input):

        sigma_input=sigma_input_list[j]

        sigma_output_matrix=np.zeros((num_iterations,1+num_var))

        for i in range(num_iterations):
            
            sigma_output_list = np.zeros(1+num_var)
                
            sigma_output_list[0] = sigma_input

            for f in range(len(freq_list)):

                signal_pure,time=signal_1D.Simulate_Signal1D(size_array,delta_time,T2,freq_list[f],False)
                
                signal_noise = signal_1D.Add_Noise1D(signal_pure,mean_set,sigma_input,False)
    
    
                coefficients = pywt.wavedec(signal_noise, wavelet, level=levels)
                approx_coef = coefficients[0]
                detail_coef_list = coefficients[1:]
                
                sigma_output_list[f+1]=Sigma5(detail_coef_list,fix=True)
    
    
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
    metrics_name_list = ['sigma{}'.format(i+1) for i in range(num_var)]

    # metrics_1D.Write_data(data_name,parameters,metrics_name_list,length_input,sigma_output_mean_list,sigma_output_std_dev_list)
    os.chdir('1D')


    os.chdir('Data')
    SIGMA = np.loadtxt(data_name+'.txt').T
    
    os.chdir('..')


    plt.figure(figsize=(10,7))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')
    
    os.chdir('Graphs')
    m0,s0 = SIGMA[0:2]
    
    a_array = np.zeros(num_var)
    b_array = np.zeros(num_var)
    
    marker_plot = ['o','s','^','>']
    color_plot = ['y','m','c','g']
    for i in range(num_var):
        freq = '{} Hz'.format(freq_list[i]*1000)
        mi,si = SIGMA[2*(i+1):2*(i+2)]




        (a,b), mcov =curve_fit(Line,m0,mi)
        a_array[i],b_array[i]=a,b
        a_,b_=1.2,0
        a_err, b_err = np.sqrt(np.diag(mcov))
        # plt.errorbar(m0,(mi-b_)/a_,yerr=si,xerr=None,marker='o',ms=20,label=freq)

        # plt.errorbar(m0,mi,yerr=si,xerr=None,marker='o',ms=20,label=freq)
        plt.errorbar(m0,mi,yerr=si,xerr=None,marker=marker_plot[i],color = color_plot[i], ms=5,label=freq)

        print(freq)
        print('\ta =',a)
        print('\ta_err =',a_err)
        
        print('\tb =',b)
        print('\tb_err =',b_err)
        
        # plt.plot(m0,Line(m0,a,b),'k--',zorder=3)


    a=1.1846438408877022
    b=0.006499950443242419
    
    
    
    plt.plot(m0,a*m0+b,'k--',zorder=3)




    # plt.xlabel(r'$\sigma_{input} (a.u.)$',fontsize=40)
    # plt.ylabel(r'$\sigma_{output} (a.u.)$',fontsize=40)
    
    plt.xlabel(r'$\sigma$')#,fontsize=60)
    plt.ylabel(r'$M$')#,fontsize=60)

    plt.legend(loc='best')#,fontsize=40)
    plt.savefig('Fig13.png')
    os.chdir('..')






