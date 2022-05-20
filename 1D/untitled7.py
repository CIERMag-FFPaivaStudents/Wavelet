#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to analyse data
"""


#Sigma_mean	Sigma_std_dev	MSE0_h_mean	MSE0_h_std_dev	MSE1_h_mean	MSE1_h_std_dev	SNR0_h_mean	SNR0_h_std_dev	SNR1_h_mean	SNR1_h_std_dev	PRD0_h_mean	PRD0_h_std_dev	PRD1_h_mean	PRD1_h_std_dev
import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mode='soft'
    sigma=2
    comp = mode + '%.d'%sigma
    data_name = 'Metrics_filter1D'
    metrics_name_list =['MSE0_h','MSE1_h','SNR0_h','SNR1_h','PRD0_h','PRD1_h']
    
    os.chdir('Data')
    data_soft_2 = np.loadtxt(data_name+'soft2.txt').T
    data_soft_5 = np.loadtxt(data_name+'soft5.txt').T

    data_hard_2 = np.loadtxt(data_name+'hard2.txt').T
    data_hard_5 = np.loadtxt(data_name+'hard5.txt').T
    
    sigmas = data_soft_2[0]
    
    
    # data = data_hard_2
    # params = 'hard2'
    # var='SNR'
    # bla=6
    # plt.figure(figsize=(12,9))
    # plt.title(var)
    # plt.errorbar(sigmas,data[bla],data[bla+1],marker='o',linestyle='--',label='noisy')
    # plt.errorbar(sigmas,data[bla+2],data[bla+3],marker='o',linestyle='none',label='smooth')
    # plt.xlabel(r'$\sigma$')
    # plt.ylabel(var)
    # plt.legend(loc='best')
    # plt.savefig(var+params+'.png')
    
    
    plt.figure(figsize=(12,9))
    params = '5'
    var='SRN'
    bla=6
    plt.title(var)
    
    plt.errorbar(sigmas,data_soft_5[bla+2],data_soft_5[bla+3],marker='o',linestyle='--',label='soft')
    plt.errorbar(sigmas,data_hard_5[bla+2],data_hard_5[bla+3],marker='o',linestyle='none',label='hard')

    # plt.errorbar(sigmas,data_soft_2[bla+2],data_soft_2[bla+3],marker='o',linestyle='--',label='soft')
    # plt.errorbar(sigmas,data_hard_2[bla+2],data_hard_2[bla+3],marker='o',linestyle='none',label='hard')


    plt.xlabel(r'$\sigma$')
    plt.ylabel(var)
    plt.legend(loc='best')
    plt.savefig(var+params+'.png')