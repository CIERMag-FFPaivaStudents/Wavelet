#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
Just to adjust the significant algorism of data tables. 

"""

import os
import numpy as np




name_list = ['FID_FFT1_000', 'FID_FFT1_010', 'FID_FFT1_025', 'FID_FFT1_075','FID_FFT2','FID_FFT3_0',
             'FID_FFT3_10', 'FID_FFT3_20', 'FID_FFT3_30', 'FID_FFT3_40', 'FID_FFT3_50', 'FID_FFT3_60',
             'FID_FFT3_70', 'FID_FFT3_80', 'FID_FFT3_90', 'FID_FFT3_100','FID_FFT4']

for name in name_list:
    
    os.chdir('Data')

    # os.chdir('Metrics_filter')
    data = np.loadtxt(name+'.txt')
    
    # os.chdir('..')
    os.chdir('clean_data')
    
    
    data_clean = open(name +'_clean.txt','w')
    
    metrics_name_list =['MSE_real','SNR_real','PRD_real','MSE_imag','SNR_imag','PRD_imag']
    num_metrics = len(metrics_name_list)
    variables_name_list=['\tParameters']
    
    variables_name_list +=metrics_name_list
    
    variables='#'
    variables+='\t'.join(variables_name_list)
    variables+='\n'
    data_clean.write(variables)
    
    m_list = []
    
    for line in data:
        line_clean = ''
        for i in range(len(line)//2):
            n,m = line[2*i:2*(i+1)]
            # m_list.append(m)
            
            sig = np.abs((np.log10(m)))
            if sig<7 and sig!=0:
                # line_clean+='\t %.7f+-%.7f'%(np.round(n,sig))
                line_clean+='\t %.7f'%(np.round(n,int(sig)))
            else:
                line_clean+='\t %.7f+-%.7f'%(n,m)
        # print(line_clean)

        line_clean+='\n'
        data_clean.write(line_clean)
    
    data_clean.close()
    
    
    # m_list = np.array(m_list)
    # log_list = np.log10(m_list)
    # bla = 10**np.round(log_list)
    os.chdir('..')
    os.chdir('..')
    