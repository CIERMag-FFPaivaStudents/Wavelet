
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is organize the plots.
"""

import os

mother_wavelet='db'
number_wavelet_list = [1,4,8]
wavelet_list = [mother_wavelet+'%d'%number_wavelet for number_wavelet in number_wavelet_list]

noise_estimator_list = ['5']
levels_dwt_list = [2,3,4,6]
mode_list = ['hard','soft']

filter_parameter_strings = ['Noise_estimator (sigma)','Thresholding mode', 'Wavelet' , 'Number of levels']

os.chdir('Data')
data_name = 'Parameters'
data = open(data_name +'.txt','w')

heading = '#Number\t'
heading += '\t'.join(filter_parameter_strings)
heading += '\n'
data.write(heading)

i=1
for noise_estimator in noise_estimator_list:
    for mode in mode_list:
        for wavelet in wavelet_list:
            for levels_dwt in levels_dwt_list:
                parameters = '{}\t{}\t'.format(i,noise_estimator)
                parameters+=mode+'\t' + wavelet
                parameters+='\t{}\n'.format(levels_dwt)
                data.write(parameters)
                i+=1


new_wavelet_list = ['sym20','coif17']
max_level_list = [5,4]
for noise_estimator in noise_estimator_list:
    for mode in mode_list:
        for wavelet in new_wavelet_list:
            max_level = max_level_list[new_wavelet_list.index(wavelet)]
            
            parameters = '{}\t{}\t'.format(i,noise_estimator)
            parameters+=mode+'\t' + wavelet
            parameters+='\t{}\n'.format(max_level)
            data.write(parameters)
            i+=1

data.close()


os.chdir('..')