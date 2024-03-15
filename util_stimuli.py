import numpy as np


def rms(x):
    '''
    Returns root mean square amplitude of x (raises ValueError if NaN).
    '''
    out = np.sqrt(np.mean(np.square(x)))
    if np.isnan(out):
        raise ValueError('rms calculation resulted in NaN')
    return out


def get_dBSPL(x, mean_subtract=True):
    '''
    Returns sound pressure level of x in dB re 20e-6 Pa (dB SPL).
    '''
    if mean_subtract:
        x = x - np.mean(x)
    out = 20 * np.log10(rms(x) / 20e-6)
    return out


def set_dBSPL(x, dBSPL, mean_subtract=True):
    '''
    Returns x re-scaled to specified SPL in dB re 20e-6 Pa.
    '''
    if mean_subtract:
        x = x - np.mean(x)
    rms_out = 20e-6 * np.power(10, dBSPL/20)
    return rms_out * x / rms(x)


def power_spectrum(x, fs, rfft=True, dBSPL=True):
    '''
    Helper function for computing power spectrum of sound wave.
    
    Args
    ----
    x (np.ndarray): input waveform (Pa)
    fs (int): sampling rate (Hz)
    rfft (bool): if True, only positive half of power spectrum is returned
    dBSPL (bool): if True, power spectrum has units dB re 20e-6 Pa
    
    Returns
    -------
    freqs (np.ndarray): frequency vector (Hz)
    power_spectrum (np.ndarray): power spectrum (Pa^2 or dB SPL)
    '''
    if rfft:
        # Power is doubled since rfft computes only positive half of spectrum
        power_spectrum = 2 * np.square(np.abs(np.fft.rfft(x) / len(x)))
        freqs = np.fft.rfftfreq(len(x), d=1/fs)
    else:
        power_spectrum = np.square(np.abs(np.fft.fft(x) / len(x)))
        freqs = np.fft.fftfreq(len(x), d=1/fs)
    if dBSPL:
        power_spectrum = 10. * np.log10(power_spectrum / np.square(20e-6)) 
    return freqs, power_spectrum


def freq2erb(freq):
    """
    Helper function converts frequency from Hz to ERB-number scale.
    Glasberg & Moore (1990, Hearing Research) equation 4. The ERB-
    number scale can be defined as the number of equivalent
    rectangular bandwidths below the given frequency (units of the
    ERB-number scale are Cams).
    """
    return 21.4 * np.log10(0.00437 * freq + 1.0)


def erb2freq(erb):
    """
    Helper function converts frequency from ERB-number scale to Hz.
    Glasberg & Moore (1990, Hearing Research) equation 4. The ERB-
    number scale can be defined as the number of equivalent
    rectangular bandwidths below the given frequency (units of the
    ERB-number scale are Cams).
    """
    return (1.0 / 0.00437) * (np.power(10.0, (erb / 21.4)) - 1.0)


def erbspace(freq_min, freq_max, num):
    """
    Helper function to get array of frequencies linearly spaced on an
    ERB-number scale.
    
    Args
    ----
    freq_min (float): minimum frequency in Hz
    freq_max (float): maximum frequency Hz
    num (int): number of frequencies (length of array)
    
    Returns
    -------
    freqs (np.ndarray): array of ERB-spaced frequencies (lowest to highest) in Hz
    """
    freqs = np.linspace(freq2erb(freq_min), freq2erb(freq_max), num=num)
    freqs = erb2freq(freqs)
    return freqs
