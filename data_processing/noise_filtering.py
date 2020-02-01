from scipy import signal


class LowPassFilter():
    def __init__(self, sample_rate=50., order=6, cutoff=8., padlen=None):
        '''
        :param sample_rate: sample rate in Hz
        :param order: butterworth filter order
        :param cutoff: desired cutoff frequency of the filter, Hz
            noises with higher frequency than this param will be filtered out
        '''
        self.padlen = padlen

        nyquist_frequency = sample_rate / 2.
        normal_cutoff = cutoff / nyquist_frequency

        (self.b, self.a) = signal.butter(N=order, Wn=normal_cutoff, btype='lowpass', analog=False)

    def denoise(self, data):
        '''
        reduce noise in the input sequence
        :param data: input sequence with shape= (timestep, channel)
        :param temporal_axis: timestep axis
        :return: same shape as input
        '''
        y = signal.filtfilt(self.b, self.a, data, axis=0, padlen=self.padlen) if self.padlen is not None \
            else signal.filtfilt(self.b, self.a, data, axis=0)
        return y
