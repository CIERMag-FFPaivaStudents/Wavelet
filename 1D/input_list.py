
#Author: Caio Oliveira
#Email: caio.dejesusoliveira@usp.br
"""
This code is to set the lit of filtering parameters.
"""

import os
import pywt

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


data.close()


os.chdir('..')



##################################################################################################3





new_filter_parameter_strings = ['Frequency(kHz)','Thresholding mode', 'Wavelet' , 'Number of levels']

mother_wavelet='coif'
number_wavelet_list = [5,10,17]
wavelet_list = [mother_wavelet+'%d'%number_wavelet for number_wavelet in number_wavelet_list]

mother_wavelet='sym'
number_wavelet_list = [8,16,20]
wavelet_list += [mother_wavelet+'%d'%number_wavelet for number_wavelet in number_wavelet_list]


size_array=2048
levels_dwt_list = [pywt.dwt_max_level(size_array, pywt.Wavelet(wavelet).dec_len) for wavelet in wavelet_list]

frequency_list = ['0','0.1','0.25','0.75']
mode_list = ['hard']


os.chdir('Data')
data_name = 'Parameters2'
data = open(data_name +'.txt','w')

heading = '#Number\t'
heading += '\t'.join(new_filter_parameter_strings)
heading += '\n'
data.write(heading)

i=25
for frequency in frequency_list:
    for mode in mode_list:
        for wavelet in wavelet_list:
            parameters = '{}\t{}\t'.format(i,frequency)
            parameters+=mode+'\t' + wavelet
            parameters+='\t{}\n'.format(levels_dwt_list[wavelet_list.index(wavelet)])
            data.write(parameters)
            i+=1



data.close()


os.chdir('..')