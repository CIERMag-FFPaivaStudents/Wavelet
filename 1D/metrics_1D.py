#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
Calculates evaluation metrics for 1D signal filtering.
"""
import numpy as np
import matplotlib.pyplot as plt
from signal_1D import Simulate_Signal1D, Add_Noise1D
import seaborn as sns
import os

def MSE_Measure(S,S_hat):
    """Measures the Mean Square Error as recommended by Aggarwal et al.
    
    Parameters
    ----------
    S: array
        Input 1D array of noiseless signal.
    S_hat: array
        Input 1D array of filtered signal.
        
    Return
    ------
    MSE: float
        Mean Square Error of the filtered signal based on the noiseless signal.
        
    References
    ----------
        AGGARWAL, Rajeev; KARAN SINGH, Jai; KUMAR GUPTA, Vijay; RATHORE,
        Sanjay; TIWARI, Mukesh; KHARE, Anubhuti. Noise Reduction of Speech Signal using
        Wavelet Transform with Modified Universal Threshold. International Journal of
        Computer Applications, [S. l.], v. 20, n. 5, p. 14–19, 2011. DOI: 10.5120/2431-3269.
        Available at: http://www.ijcaonline.org/volume20/number5/pxc3873269.pdf.
    """

    diff = np.subtract(S,S_hat)
    d = np.sum(np.power(diff,2))
    MSE = d/len(S)
    return MSE


def SNR_Measure1D(S,S_hat):
    """Measure of signal to noise ratio in 1D signals as defined in Mitra et al.
    
    Parameters
    ----------
    S: array
        Input 1D array of noiseless signal.
    S_hat: array
        Input 1D array of filtered signal.
        
    Return
    ------
    SNR: float
        Signal to noise ratio of the filtered signal.
        
    References
    ----------
        MITRA, D.; SHAHJALAL, Muhammad; KIBER, Adnan. Comparative Study on
        Thresholding Techniques of Discrete Wavelet Transform (DWT) to De-noise
        Corrupted ECG Signals. In: INTERNATIONAL JOURNAL OF ENGINEERING AND
        COMPUTER SCIENCE 2014, Annals [...]. [s.l: s.n.] p. 7601–7604. Available at:
        www.ijecs.in
    """
    power_signal = np.sum(np.power(S,2))
    noise_est = np.subtract(S,S_hat)
    power_noise=np.sum(np.power(noise_est,2))
    SNR = 10*np.log10(power_signal/power_noise)
    return SNR


def PRD_Measure(S,S_hat):
    """Measure of percent root mean square difference as defined in Mitra et al.
    
    Parameters
    ----------
    S: array
        Input 1D array of noiseless signal.
        
    S_hat: array
        Input 1D array of filtered signal.
        
    Return
    ------
    PRD: float
        Percent root mean square difference of the noiseless signal and the filtered signal as defined in Mitra et al.

    References
    ----------
        MITRA, D.; SHAHJALAL, Muhammad; KIBER, Adnan. Comparative Study on
        Thresholding Techniques of Discrete Wavelet Transform (DWT) to De-noise
        Corrupted ECG Signals. In: INTERNATIONAL JOURNAL OF ENGINEERING AND
        COMPUTER SCIENCE 2014, Annals [...]. [s.l: s.n.] p. 7601–7604. Available at:
        www.ijecs.in
    """
    noise_est = np.subtract(S,S_hat)
    power_signal=np.sum(np.power(S,2))
    power_noise=np.sum(np.power(noise_est,2))
    PRD=np.sqrt(power_noise/power_signal)
    return PRD


def Write_data(data_name,parameters,metrics_name_list,length_input,metrics_mean_list,metrics_std_dev_list,folder=''):
    """Writes the metrics of a given simulation in a datasheet in .txt format.
    
    Parameters
    ----------
    data_name: string
        Desired name of data_sheet archive.
        
    metrics_name_list: list of strings
        Names of used metrics
        
    legnth_input: int
        Size of sigma_input_list.
    
    metrics_mean_list: array
        Array with the mean values of each metric in each input.

    metrics_std_dev_list: array
        Array with the standard deviation of each metric in each input.
    
    Return
    ------

    References
    ----------
    """
    os.chdir('Data'+folder)
    
    data = open(data_name +'.txt','w')
    os.chdir('..')
    num_metrics = len(metrics_name_list)
    variables_name_list=['Sigma_mean','Sigma_std_dev']

    for metrics_name in metrics_name_list:
        variables_name_list.append(metrics_name+'_mean')
        variables_name_list.append(metrics_name+'_std_dev')
    
    data.write(parameters+'\n')

    variables='#'
    variables+='\t'.join(variables_name_list)
    variables+='\n'
    data.write(variables)
    data_line=''

    for j in range(length_input):
        for k in range(num_metrics+1):
          m,s=metrics_mean_list[j][k],metrics_std_dev_list[j][k]
          data_line+='{}\t{}'.format(m,s)
          data_line+= '\t'
        data_line+='\n'

    data.write(data_line)
    data.close()

    os.chdir('..')
    return None


def Show_data(data_name,figsize,metrics_name_list,path='Graphs'):
    """Shows the metrics of a given simulation in plotted graph for each metric.
    
    Parameters
    ----------
    data_name: string
        Desired name of data_sheet archive.
        
    figsize: tuple
        Size of plt.figure. Width, height in inches.
        
    metrics_name_list: list of strings
        Names of the used metrics

    Return
    ------

    References
    ----------
    """


    os.chdir('1D/Data')
    Metrics = np.loadtxt(data_name +'.txt').T
    os.chdir('..')

    plt.figure(figsize=figsize)
    sns.set()
    sns.set_style('ticks')
    sns.set_context('talk')

    os.chdir(path)
    m0,s0 = Metrics[0:2]
    num_metrics = len(metrics_name_list)
    for i in range(num_metrics):
        plt.clf()
        mi,si = Metrics[2*(i+1):2*(i+2)]
        plt.errorbar(m0,mi,yerr=si,color='k',marker='o',linestyle='none')
        plt.xlabel('$\sigma_{input} (a.u.)$')
        plt.ylabel(metrics_name_list[i])
        plt.savefig(metrics_name_list[i]+'.png')
    os.chdir('..')
    os.chdir('..')
    return None




if __name__ == '__main__':

    delta_time=0.5
    T2=100
    freq=0.25 
    size_array=2048
    mean=0
    std_dev=0.01


    num_iterations=1

    sigma_input_list=np.arange(0,0.6,0.01)

    length_input=len(sigma_input_list)

    num_metrics=3

    metrics_mean_list=np.zeros((length_input,num_metrics+1))
    metrics_std_dev_list=np.zeros((length_input,num_metrics+1))


    for j in range(length_input):
        sigma_input=sigma_input_list[j]
        
        metrics_matrix=np.zeros((num_iterations,num_metrics+1))
        
        for i in range(num_iterations):
            signal_pure,time=Simulate_Signal1D(size_array,delta_time,T2,freq)
            
            signal_noise = Add_Noise1D(signal_pure,mean,sigma_input)


            metrics_list = np.zeros(num_metrics+1)

            metrics_list[0] = sigma_input


            MSE=MSE_Measure(signal_pure, signal_noise)
            metrics_list[1] = MSE

            SNR = SNR_Measure1D(signal_pure,signal_noise)
            metrics_list[2] = SNR

            PRD = PRD_Measure(signal_pure,signal_noise)
            metrics_list[3] = PRD


            metrics_matrix[i]=metrics_list

        metrics_matrix = metrics_matrix.T

        metrics_mean=[metrics_list.mean() for metrics_list in metrics_matrix]
        metrics_mean = np.array(metrics_mean).T
        metrics_mean_list[j]=metrics_mean

        metrics_std_dev=[np.std(metrics_list) for metrics_list in metrics_matrix]
        metrics_std_dev = np.array(metrics_std_dev).T
        metrics_std_dev_list[j]=metrics_std_dev

    data_name = 'Metrics_1D'
    metrics_name_list =['MSE','SNR','PRD']
    parameters  = ''
    
    Write_data(data_name,parameters,metrics_name_list,length_input,metrics_mean_list,metrics_std_dev_list)

    figsize=(12,9)
    Show_data(data_name,figsize,metrics_name_list)