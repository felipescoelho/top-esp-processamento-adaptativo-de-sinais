#!/usr/bin/python
# -*- coding: utf-8 -*-


#----------------------------------------------------------------------------------------------#
# LIBRARIES
#----------------------------------------------------------------------------------------------#

import numpy as np # To work with arrays and matrices


#----------------------------------------------------------------------------------------------#
# MAIN CODE
#----------------------------------------------------------------------------------------------#

def padding_array(x, N, padding_type = 0):

    '''
        Parameters
        ----------

        x:
        N:
        padding_type:


        Returns
        -------
    '''

    x = np.append(np.ones(N + 1) if padding_type else np.zeros(N + 1), x) # 0/1 padding the vector

    return np.array([x[k + N : k - 1 : -1] for k in range(1, x.size - N)]) # Creating the input of order N

#----------------------------------------------------------------------------------------------#

def mse(x, d, dB = 1):

    '''

        Parameters
        ----------


        Returns
        -------

    '''

    error = np.zeros(len(d))

    for x_i in x:
        error += (d - x_i) ** 2
 
    return 10 * np.log10(error / len(x)) if dB else error / len(x)


#----------------------------------------------------------------------------------------------#
# MAIN SCRIPT
#----------------------------------------------------------------------------------------------#

if __name__ == '__main__':

    pass


#----------------------------------------------------------------------------------------------#
# End of File (EOF)
#----------------------------------------------------------------------------------------------#


# Adding noise using target SNR

# Set a target SNR
# target_snr_db = 20
# Calculate signal power and convert to dB 
# sig_avg_watts = np.mean(x_watts)
# sig_avg_db = 10 * np.log10(sig_avg_watts)
# Calculate noise according to [2] then convert to watts
# noise_avg_db = sig_avg_db - target_snr_db
# noise_avg_watts = 10 ** (noise_avg_db / 10)
# Generate an sample of white noise
# mean_noise = 0
# noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))