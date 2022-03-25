#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
Calculates evaluation metrics for 1D signal filtering.
"""
import numpy as np
import matplotlib.pyplot as plt
from signal_1D import Simulate_Signal1D, Add_Noise1D
import seaborn as sns


def MSE_Measure(S,S_hat):
    diff = np.subtract(S,S_hat)
    d = np.sum(np.power(diff,2))
    MSE = d/len(S)
    return MSE


def SNR_Measure1D(S,S_hat):
    power_signal = np.sum(np.power(S,2))
    noise_est = np.subtract(S,S_hat)
    power_noise=np.sum(np.power(noise_est,2))
    # print(power_noise)
    # if power_noise==0:
    #     return None
    # else:
    SNR = 10*np.log10(power_signal/power_noise)
    return SNR


def PRD_Measure(S,S_hat):
    noise_est = np.subtract(S,S_hat)
    power_signal=np.sum(np.power(S,2))
    power_noise=np.sum(np.power(noise_est,2))
    PRD=np.sqrt(power_noise/power_signal)
    return PRD


if __name__ == '__main__':
    delta_time=0.5
    T2=100
    freq=0.25 
    size_array=2048
    mean=0
    std_dev=0.01
    
    
    num_iterations=1000
    
    
    sigma_input_list=np.arange(0,0.6,0.01)

    length_input=len(sigma_input_list)

    num_metrics=3

    # metrics_mean_list=np.zeros((length_input,num_metrics+1))
    # metrics_std_dev_list=np.zeros((length_input,num_metrics+1))
    
    

    # for j in range(length_input):
    #     # continue
    #     sigma_input=sigma_input_list[j]
        
    #     metrics_matrix=np.zeros((num_iterations,num_metrics+1))
        
    #     for i in range(num_iterations):
    #         # print(j,i)
    #         signal_pure,time=Simulate_Signal1D(size_array,delta_time,T2,freq)
            
    #         signal_noise = Add_Noise1D(signal_pure,mean,sigma_input)

            

            

    #         metrics_list = np.zeros(num_metrics+1)
            
    #         metrics_list[0] = sigma_input
            
            

    #         MSE=MSE_Measure(signal_pure, signal_noise)
    #         metrics_list[1] = MSE
            
    #         SNR = SNR_Measure1D(signal_pure,signal_noise)
    #         metrics_list[2] = SNR
            
    #         PRD = PRD_Measure(signal_pure,signal_noise)
    #         metrics_list[3] = PRD
            

    #         metrics_matrix[i]=metrics_list

    #     # continue
    #     metrics_matrix = metrics_matrix.T

    #     metrics_mean=[metrics_list.mean() for metrics_list in metrics_matrix]
    #     metrics_mean = np.array(metrics_mean).T
    #     metrics_mean_list[j]=metrics_mean
    
    #     metrics_std_dev=[np.std(metrics_list) for metrics_list in metrics_matrix]
    #     metrics_std_dev = np.array(metrics_std_dev).T
    #     metrics_std_dev_list[j]=metrics_std_dev



    # # Escrevendo na tabela
    # Nome='Metricas'
    # tabela = open(Nome+'.txt','w')
    
    
    # nomes_metricas=['Sigma_mean','Sigma_std_dev','MSE_mean','MSE_std_dev','SNR_mean','SNR_std_dev','PRD_mean','PRD_std_dev']
    # var='#'
    # var+='\t'.join(nomes_metricas)
    # var+='\n'
    # tabela.write(var)
    
    # pontos=''
    # for j in range(length_input):
    #     for k in range(num_metrics+1):
    #       m,s=metrics_mean_list[j][k],metrics_std_dev_list[j][k]
    #       pontos+='{}\t{}'.format(m,s)
    #       pontos+= '\t'
    #     pontos+='\n'
    # tabela.write(pontos)
    
    # tabela.close()
    
    
    

    
    Metrics = np.loadtxt('Metricas.txt').T
    
    symbols=['h','*','o','^','s','p','H']
    plt.figure(figsize=(12,9))
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')
    
    nomes_metricas =['MSE','SNR','PRD']
    m0,s0 = Metrics[0:2]
    for i in range(num_metrics):
        plt.clf()
        mi,si = Metrics[2*(i+1):2*(i+2)]
        plt.errorbar(m0,mi,yerr=si,color='k',marker='o',linestyle='none')
        plt.xlabel('$\sigma_{input} (a.u.)$')
        plt.ylabel(nomes_metricas[i])
        plt.savefig('Grafico '+nomes_metricas[i]+'.png')

        