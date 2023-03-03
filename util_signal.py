import os
import sys
import pdb
import numpy as np
import tensorflow as tf
import scipy.signal


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


def tf_rms(x, axis=-1, keepdims=True):
    """
    Compute root mean square amplitude of a tensor.
    
    Args
    ----
    x (tensor): tensor for which root mean square amplitude is computed
    axis (int or list): axis along which to compute mean
    keepdims (bool): specifies if mean should keep collapsed dimension(s)
    
    Returns
    -------
    out (tensor): root mean square amplitude of x
    """
    out = tf.math.sqrt(
        tf.math.reduce_mean(
            tf.math.square(x),
            axis=axis,
            keepdims=keepdims))
    return out


def tf_get_dbspl(x, mean_subtract=True, axis=-1, keepdims=True):
    """
    Get sound pressure level in dB re 20e-6 Pa (dB SPL) of a tensor.
    
    Args
    ----
    x (tensor): tensor for which sound pressure level is computed
    mean_subtract (bool): if True, x is first de-meaned
    axis (int or list): axis along which to measure RMS amplitudes
    keepdims (bool): specifies if mean should keep collapsed dimension(s)
    
    Returns
    -------
    out (tensor): sound pressure level of x in dB re 20e-6 Pa
    """
    if mean_subtract:
        x = x - tf.math.reduce_mean(x, axis=axis, keepdims=True)
    out = 20.0 * tf.math.divide(
        tf.math.log(tf_rms(x, axis=axis, keepdims=keepdims) / 20e-6),
        tf.math.log(tf.constant(10, dtype=x.dtype)))
    return out


def tf_set_dbspl(x, dbspl, mean_subtract=True, axis=-1, strict=True):
    """
    Set sound pressure level in dB re 20e-6 Pa (dB SPL) of a tensor.
    
    Args
    ----
    x (tensor): tensor for which sound pressure level is set
    dbspl (tensor): desired sound pressure level in dB re 20e-6 Pa
    mean_subtract (bool): if True, x is first de-meaned
    axis (int or list): axis along which to measure RMS amplitudes
    strict (bool): if True, an error will be raised if x is silent;
        if False, silent signals will be returned as-is
    
    Returns
    -------
    out (tensor): sound pressure level of x in dB re 20e-6 Pa
    """
    if mean_subtract:
        x = x - tf.math.reduce_mean(x, axis=axis, keepdims=True)
    rms_new = 20e-6 * tf.math.pow(10.0, dbspl / 20.0)
    rms_old = tf_rms(x, axis=axis, keepdims=True)
    if strict:
        tf.debugging.assert_none_equal(
            rms_old,
            tf.zeros_like(rms_old),
            message="Failed to set dB SPL of all-zero signal")
        out = tf.math.multiply(rms_new / rms_old, x)
    else:
        out = tf.where(
            tf.math.equal(rms_old, tf.zeros_like(rms_old)),
            x,
            tf.math.multiply(rms_new / rms_old, x))
    return out


class rnn_biquad_digital_filter():
    """
    Tensorflow RNN cell implementing a digital biquad filter.
    """
    def __init__(self, b, a, dtype=tf.float32):
        """
        Initialize RNN cell for digital biquad filter
        """
        assert b.shape[0] == 3, "numerator coefs must have shape [3, N]"
        assert a.shape[0] == 3, "denominator coefs must have shape [3, N]"
        b0, b1, b2 = b.reshape([3, -1])
        a0, a1, a2 = a.reshape([3, -1])
        assert np.array_equal(a0, np.ones_like(a0)), "coefs must be normalized"
        
        N = a0.shape[0]
        self.state_size = tf.TensorShape([3, N])
        
        A = np.zeros([3, 3, N])
        A[0, 0, :] = -a1
        A[0, 1, :] = 1
        A[1, 0, :] = -a2
        A[1, 2, :] = 1
        self.A = tf.constant(A, dtype=dtype)
        
        K = np.zeros([1, 3, N])
        K[0, 0, :] = 1
        self.K = tf.constant(K, dtype=dtype)
        
        B = np.zeros([3, 1, N])
        B[0, 0, :] = b0
        B[1, 0, :] = b1
        B[2, 0, :] = b2
        self.B = tf.constant(B, dtype=dtype)
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Define initial state of filter: zeros with shape [batch, 3, N]
        """
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        if batch_size is None or dtype is None:
            raise ValueError(
                "batch_size and dtype cannot be None while constructing initial state: "
                "batch_size={}, dtype={}".format(batch_size, dtype))
        initial_state = tf.zeros([batch_size] + self.state_size.as_list(), dtype=dtype)
        return initial_state
    
    def call(self, x, w):
        """
        Parallelized implementation of direct form II digital biquad filter
        difference equation (RNN state vector stores w[n], w[n-1], w[n-2]):
            w[n] = x[n] - (a1 * w[n-1]) - (a2 * w[n-2])
            y[n] = (b0 * w[n]) + (b1 * w[n-1]) + (b2 * w[n-2])
        """
        w_next = tf.math.add(
            tf.transpose(
                tf.linalg.matmul(
                    tf.transpose(w[0], perm=[2, 0, 1]),
                    tf.transpose(self.A, perm=[2, 0, 1])
                ),
                perm=[1, 2, 0]
            ),
            tf.transpose(
                tf.linalg.matmul(
                    tf.transpose(x[:, tf.newaxis, :], perm=[2, 0, 1]),
                    tf.transpose(self.K, perm=[2, 0, 1])
                ),
                perm=[1, 2, 0]
            )
        )
        y = tf.squeeze(
            tf.transpose(
                tf.linalg.matmul(
                    tf.transpose(self.B, perm=[2, 1, 0]),
                    tf.transpose(w_next, perm=[2, 1, 0])
                ),
                perm=[2, 1, 0]
            ),
            axis=1
        )
        return (y, w_next)


def get_gammatone_filter_coefs(fs,
                               cfs,
                               EarQ=9.2644,
                               minBW=24.7,
                               order=1):
    """
    Based on `MakeERBFilters.m` and `ERBFilterBank.m`
    from Malcolm Slaney's Auditory Toolbox (1998).
    """
    T = 1 / fs
    ERB = ((cfs / EarQ) ** order + minBW ** order) ** (1/order)
    B = 1.019 * 2 * np.pi * ERB
    A0 = T * np.ones_like(cfs)
    A2 = 0 * np.ones_like(cfs)
    B0 = 1 * np.ones_like(cfs)
    B1 = -2 * np.cos(2 * cfs * np.pi * T) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)
    
    tmp0 = 2 * T * np.cos(2 * cfs * np.pi * T) / np.exp(B * T)
    tmp1 = T * np.sin(2 * cfs * np.pi * T) / np.exp(B * T)
    A11 = -(tmp0 + 2 * np.sqrt(3 + 2 ** 1.5) * tmp1) / 2;
    A12 = -(tmp0 - 2 * np.sqrt(3 + 2 ** 1.5) * tmp1) / 2;
    A13 = -(tmp0 + 2 * np.sqrt(3 - 2 ** 1.5) * tmp1) / 2;
    A14 = -(tmp0 - 2 * np.sqrt(3 - 2 ** 1.5) * tmp1) / 2;

    tmp2 = np.exp(4 * 1j * cfs * np.pi * T)
    tmp3 = 2 * np.exp(-(B * T) + 2 * 1j * cfs * np.pi * T) * T
    tmp4 = np.cos(2 * cfs * np.pi * T)
    tmp5 = np.sin(2 * cfs * np.pi * T)
    gain = np.abs(
        (-2 * tmp2 * T + tmp3 * (tmp4 - np.sqrt(3 - 2 ** (3 / 2)) * tmp5)) * \
        (-2 * tmp2 * T + tmp3 * (tmp4 + np.sqrt(3 - 2 ** (3 / 2)) * tmp5)) * \
        (-2 * tmp2 * T + tmp3 * (tmp4 - np.sqrt(3 + 2 ** (3 / 2)) * tmp5)) * \
        (-2 * tmp2 * T + tmp3 * (tmp4 + np.sqrt(3 + 2 ** (3 / 2)) * tmp5)) / \
        (-2 / np.exp(2 * B * T) - 2 * tmp2 + 2 * (1 + tmp2) / np.exp(B * T)) ** 4
    )
    
    filter_coefs = [
        {
            'b': np.array([A0, A11, A2]) / gain,
            'a': np.array([B0, B1, B2])
        },
        {
            'b': np.array([A0, A12, A2]),
            'a': np.array([B0, B1, B2])
        },
        {
            'b': np.array([A0, A13, A2]),
            'a': np.array([B0, B1, B2])
        },
        {
            'b': np.array([A0, A14, A2]),
            'a': np.array([B0, B1, B2])
        },
    ]
    return filter_coefs


def scipy_gammatone_filterbank(x, filter_coefs):
    """
    Convert signal waveform `x` to set of subbands `x_subbands`
    using scipy.signal.lfilter and the gammatone filterbank
    instantiated by `filter_coefs`.
    """
    if len(x.shape) == 1:
        x_subbands = x[np.newaxis, np.newaxis, :]
    elif len(x.shape) == 2:
        x_subbands = x[:, np.newaxis, :]
    else:
        raise ValueError("Expected input shape [time] or [batch, time]")
    n_subbands = filter_coefs[0]['b'].shape[-1]
    x_subbands = np.tile(x_subbands, [1, n_subbands, 1])
    for fc in filter_coefs:
        for itr_subbands in range(n_subbands):
            x_subbands[:, itr_subbands, :] = scipy.signal.lfilter(
                fc['b'][:, itr_subbands],
                fc['a'][:, itr_subbands],
                x_subbands[:, itr_subbands, :],
                axis=-1)
    if len(x.shape) == 1:
        x_subbands = x_subbands[0]
    return x_subbands


def rnn_gammatone_filterbank(x,
                             fs,
                             cfs=None,
                             min_cf=125.0,
                             max_cf=8e3,
                             num_cf=50,
                             kwargs_filter_coefs={},
                             dtype=tf.float32):
    """
    """
    if len(x.shape) == 1:
        x = x[tf.newaxis, :, tf.newaxis]
    elif len(x.shape) == 2:
        x = x[:, :, tf.newaxis]
    elif len(x.shape) > 3:
        raise ValueError("Input dimensions should be: [batch, time, channels]")
    if cfs is None:
        cfs = erbspace(min_cf, max_cf, num_cf)
    filter_coefs = get_gammatone_filter_coefs(fs, cfs, **kwargs_filter_coefs)
    list_rnn_cells = [rnn_biquad_digital_filter(**fc) for fc in filter_coefs]
    gammatone_rnn_layer = tf.keras.layers.RNN(
        list_rnn_cells,
        return_sequences=True,
        return_state=False,
        go_backwards=False,
        stateful=False,
        unroll=False,
        time_major=False,
        dtype=dtype)
    y = gammatone_rnn_layer(x)
    container = {
        'fs': fs,
        'cfs': cfs,
    }
    return y, container


def get_gammatone_impulse_responses(fs,
                                    fir_dur,
                                    cfs,
                                    EarQ=9.2644,
                                    minBW=24.7,
                                    order=1):
    """
    """
    impulse = np.zeros(int(fir_dur * fs))
    impulse[0] = 1
    impulse_responses = np.tile(impulse[np.newaxis, :], [len(cfs), 1])
    filter_coefs = get_gammatone_filter_coefs(
        fs,
        cfs,
        EarQ=9.2644,
        minBW=24.7,
        order=1)
    for fc in filter_coefs:
        for itr_channel in range(len(cfs)):
            impulse_responses[itr_channel, :] = scipy.signal.lfilter(
                fc['b'][:, itr_channel],
                fc['a'][:, itr_channel],
                impulse_responses[itr_channel, :])
    return impulse_responses


def fir_gammatone_filterbank(x,
                             fs,
                             fir_dur,
                             cfs=None,
                             min_cf=125.0,
                             max_cf=8e3,
                             num_cf=50,
                             kwargs_filter_coefs={},
                             padding='SAME'):
    """
    """
    if len(x.shape) == 1:
        x = x[tf.newaxis, :, tf.newaxis]
    elif len(x.shape) == 2:
        x = x[:, :, tf.newaxis]
    elif len(x.shape) > 3:
        raise ValueError("Input dimensions should be: [batch, time, channels]")
    if cfs is None:
        cfs = erbspace(min_cf, max_cf, num_cf)
    fir = get_gammatone_impulse_responses(fs, fir_dur, cfs, **kwargs_filter_coefs)
    fir_kernel = np.zeros([2 * fir.shape[1] - 1, 1, fir.shape[0]])
    fir_kernel[:fir.shape[1], 0, :] = fir[:, ::-1].T
    fir_kernel_tensor = tf.constant(fir_kernel, dtype=x.dtype)
    y = tf.nn.convolution(x, filters=fir_kernel_tensor, padding=padding)
    y = tf.transpose(y, perm=[0, 2, 1])
    container = {
        'fs': fs,
        'fir_dur': fir_dur,
        'cfs': cfs,
        'fir': fir,
        'fir_kernel': fir_kernel,
    }
    return y, container


def get_roex_transfer_function(p, r, cf):
    """
    Returns the transfer function for a rounded exponential filter.
    Signals, Sound, and Senation / Hartmann (1998) pages 247-248.
    """
    g = lambda f: np.abs((f - cf) / cf)
    roex = lambda f: r + (1.0 - r) * (1.0 + p * g(f)) * np.exp(-p * g(f))
    return roex


def make_roex_filters(signal_length,
                      sr,
                      cfs=None,
                      bws=None,
                      dc_ramp_cutoff=30):
    """
    Function builds a filterbank of rounded exponential bandpass filters.
    
    Args
    ----
    signal_length (int): length of signal (in samples) that filters will be applied to
    sr (int): sampling rate (Hz)
    cfs (np.ndarray): filterbank center frequencies (Hz)
    bws (np.ndarray): filterbank bandwidths (Hz)
    dc_ramp_cutoff (float): filterbank is multiplied by a ramp that goes from 0 to 1 between
        0 and `dc_ramp_cutoff` Hz to remove DC component (only applied if dc_ramp_cutoff > 0)
    
    Returns
    -------
    filts (np.ndarray): roex filterbank, an array of floats with shape [num_cf, num_freq]
    freqs (np.ndarray): frequency vector (Hz)
    """
    assert (cfs is not None) and (bws is not None), "cfs and bws are required arguments"
    assert cfs.shape == bws.shape, "cfs and bws must be arrays with the same shape"
    num_cf = cfs.shape[0]
    # Setup frequency vector and initialize filter array
    if np.remainder(signal_length, 2) == 0: # even length
        num_freq = int(signal_length // 2) + 1
        max_freq = sr / 2
    else: # odd length
        num_freq = int((signal_length - 1) // 2) + 1
        max_freq = sr * (signal_length - 1) / 2 / signal_length
    freqs = np.linspace(0, max_freq, num_freq)
    filts = np.zeros((num_cf, num_freq))
    # Build the roex filterbank
    for fidx, (cf, bw) in enumerate(zip(cfs, bws)):
        p = (4 * cf) / bw # Convert bw to dimensionless p for roex transfer function
        r = 0
        roex = get_roex_transfer_function(p, r, cf)
        filts[fidx, :] = roex(freqs)
        if cf + bw / 2 > sr / 2:
            msg = "WARNING: High ERB cutoff of filter with cf={:.1f}Hz exceeds Nyquist"
            print(msg.format(cf))
        if cf - bw / 2 <= 0:
            msg = "WARNING: Low ERB cutoff of filter with cf={:.1f}Hz is not strictly positive"
            print(msg.format(cf))
    # Multiply the filterbank by a ramp to eliminate gain at DC component
    if dc_ramp_cutoff > 0:
        print("Including ramp below {}Hz to eliminate DC offset in roex filterbank".format(dc_ramp_cutoff))
        dc_ramp = np.ones([1, freqs.shape[0]])
        dc_ramp[:, freqs < dc_ramp_cutoff] = freqs[freqs < dc_ramp_cutoff] / dc_ramp_cutoff
        filts = filts * dc_ramp
    return filts, freqs


def roex_filterbank(x,
                    sr,
                    num_cf=50, 
                    min_cf=125.0,
                    max_cf=8e3,
                    species=3,
                    bandwidth_scale_factor=1.0,
                    cfs=None,
                    bws=None,
                    dc_ramp_cutoff=30):
    """
    """
    # Prepare list of CFs (default is spaced linearly on an ERB scale)
    if cfs is None:
        assert max_cf <= sr/2, "max_cf must be below Nyquist frequency"
        assert min_cf < max_cf, "min_cf must be below max_cf?"
        cfs = erbspace(min_cf, max_cf, num_cf)
    else:
        cfs = np.array(cfs)
    msg = "invalid cfs shape {} with n={}".format(cfs.shape, num_cf)
    assert (cfs.shape[0] == num_cf) and (len(cfs.shape) == 1), msg
    # Prepare list of filter BWs (default is Glasberg & Moore, 1990)
    if bws is None:
        if species == 1: # cat QERB values from Shera et al. (PNAS 2002)
            QERB = np.power((cfs / 1000), 0.37) * 5.0
        elif species == 2: # human QERB values from Shera et al. (PNAS 2002)
            QERB = np.power((cfs / 1000), 0.3) * 12.7
        elif species == 3: # human QERB values from Glasberg & Moore (Hear. Res. 1990)
            QERB = cfs / (24.7 * (1 + 4.37 * (cfs / 1000)))
        else:
            raise ValueError("species must be 1=cat, 2=human (Shera), or 3=human (G&M)")
        bws = bandwidth_scale_factor * cfs / QERB
    else:
        bws = np.array(bws)
    msg = "invalid bws shape {} with n={}".format(bws.shape, num_cf)
    assert (bws.shape[0] == num_cf) and (len(bws.shape) == 1), msg
    # Prepare input signal and apply roex filterbank in frequency domain
    if len(x.shape) == 1:
        x = x[tf.newaxis, :]
    elif len(x.shape) > 2:
        raise ValueError("Input dimensions should be: [batch, time]")
    filts, freqs = make_roex_filters(
        x.shape[-1],
        sr,
        cfs=cfs,
        bws=bws,
        dc_ramp_cutoff=dc_ramp_cutoff)
    rfft_x = tf.keras.layers.Lambda(tf.signal.rfft, dtype=tf.float32)(x)
    rfft_x = rfft_x[:, tf.newaxis, :]
    rfft_filts = tf.constant(filts, dtype=rfft_x.dtype)
    rfft_filts = rfft_filts[tf.newaxis, :, :]
    rfft_y = tf.math.multiply(rfft_x, rfft_filts)
    y = tf.keras.layers.Lambda(tf.signal.irfft)(rfft_y)
    container = {
        'sr': np.array(sr),
        'cfs': cfs,
        'bws': bws,
        'freqs': freqs,
        'filts': filts,
    }
    return y, container


def get_half_cosine_transfer_function(lo=None, hi=None, cf=None, bw_erb=None):
    """
    Returns the transfer function for a half-cosine filter, which
    can be specified by either a low and high frequency cutoff or
    a center frequency (in Hz) and bandwidth (in ERB).
    """
    msg = "Specify half cosine filter using either (lo, hi) XOR (cf, bw_erb)"
    if (lo is not None) and (hi is not None):
        assert (cf is None) and (bw_erb is None), msg
        lo_erb = freq2erb(lo)
        hi_erb = freq2erb(hi)
        cf_erb = (lo_erb + hi_erb) / 2
        bw_erb = hi_erb - lo_erb
    elif (cf is not None) and (bw_erb is not None):
        assert (lo is None) and (hi is None), msg
        cf_erb = freq2erb(cf)
        lo_erb = cf_erb - (bw_erb / 2)
        hi_erb = cf_erb + (bw_erb / 2)
    else:
        raise ValueError(msg)
    def half_cosine_transfer_function(f):
        f_erb = freq2erb(f)
        out = np.zeros_like(f_erb)
        IDX = np.logical_and(f_erb > lo_erb, f_erb < hi_erb)
        out[IDX] = np.cos(np.pi * (f_erb[IDX] - cf_erb) / bw_erb)
        return out
    return half_cosine_transfer_function


def make_half_cosine_filters(signal_length,
                             sr,
                             list_lo=None,
                             list_hi=None):
    """
    Function builds a filterbank of half-cosine bandpass filters.
    
    Args
    ----
    signal_length (int): length of signal (in samples) that filters will be applied to
    sr (int): sampling rate (Hz)
    list_lo (np.ndarray): filterbank low frequency cutoffs (Hz)
    list_hi (np.ndarray): filterbank high frequency cutoffs (Hz)
    
    Returns
    -------
    filts (np.ndarray): half-cosine filterbank, an array of floats with shape [num_cf, num_freq]
    freqs (np.ndarray): frequency vector (Hz)
    """
    assert (list_lo is not None) and (list_hi is not None), "list_lo and list_hi are required args"
    assert list_lo.shape == list_hi.shape, "list_lo and list_hi must be arrays with the same shape"
    num_cf = list_lo.shape[0]
    # Setup frequency vector and initialize filter array
    if np.remainder(signal_length, 2) == 0: # even length
        num_freq = int(signal_length // 2) + 1
        max_freq = sr / 2
    else: # odd length
        num_freq = int((signal_length - 1) // 2) + 1
        max_freq = sr * (signal_length - 1) / 2 / signal_length
    freqs = np.linspace(0, max_freq, num_freq)
    filts = np.zeros((num_cf, num_freq))
    # Build the half-cosine filterbank
    for fidx, (lo, hi) in enumerate(zip(list_lo, list_hi)):
        assert lo < hi, "low frequency cutoff must be < high frequency cutoff"
        halfcos = get_half_cosine_transfer_function(lo=lo, hi=hi)
        filts[fidx, :] = halfcos(freqs)
    return filts, freqs


def half_cosine_filterbank(x,
                           sr,
                           num_cf=50,
                           min_lo=20.0,
                           max_hi=10e3):
    """
    """
    # Prepare list of CFs / BWs (default is spaced linearly on an ERB scale)
    list_cutoffs = erbspace(min_lo, max_hi, num_cf + 2)
    list_lo = list_cutoffs[:-2]
    list_hi = list_cutoffs[2:]
    cfs = list_cutoffs[1:-1]
    bws = list_hi - list_lo
    bws_erb = freq2erb(list_hi) - freq2erb(list_lo)
    # Prepare input signal and apply half-cosine filterbank in frequency domain
    if len(x.shape) == 1:
        x = x[tf.newaxis, :]
    elif len(x.shape) > 2:
        raise ValueError("Input dimensions should be: [batch, time]")
    filts, freqs = make_half_cosine_filters(
        x.shape[-1],
        sr,
        list_lo=list_lo,
        list_hi=list_hi)
    rfft_x = tf.keras.layers.Lambda(tf.signal.rfft, dtype=tf.float32)(x)
    rfft_x = rfft_x[:, tf.newaxis, :]
    rfft_filts = tf.constant(filts, dtype=rfft_x.dtype)
    rfft_filts = rfft_filts[tf.newaxis, :, :]
    rfft_y = tf.math.multiply(rfft_x, rfft_filts)
    y = tf.keras.layers.Lambda(tf.signal.irfft)(rfft_y)
    container = {
        'sr': np.array(sr),
        'cfs': cfs,
        'bws': bws,
        'bws_erb': bws_erb,
        'list_lo': list_lo,
        'list_hi': list_hi,
        'freqs': freqs,
        'filts': filts,
    }
    return y, container


def nnresample_poly_filter(up, down, beta=5.0, window_length=16001, nnshift=True, legacy=False):
    """
    Builds an anti-aliasing lowpass filter with cutoff approximately equal
    to (1/2) * INITIAL_SAMPLING_RATE * up / down.
    Null-on-Nyquist anti-aliasing filter implementation based on:
    https://github.com/jthiem/nnresample/blob/master/nnresample/nnresample.py
    
    Args
    ----
    up (int): upsampling factor
    down (int): downsampling factor
    beta (float): Kaiser window shape parameter
    window_length (int): finite impulse response window length
    nnshift (bool): shift anti-aliasing filter cutoff to move null-on-Nyquist
    legacy (bool): construct lowpass filter from sinc function + kaiser window
    
    Returns
    -------
    shifted_filt (np.array of shape [window_length]): filter impulse response
    """
    # Ensure the specifed upsampling and downsampling factors are efficient
    # (rational approximation to save computation time on really long signals)
    # *** this code block was stolen from scipy.signal.resample_poly ***
    greatest_common_divisor = np.gcd(up, down)
    up = up // greatest_common_divisor
    down = down // greatest_common_divisor
    if legacy:
        # Construct lowpass-filter with sinc function and kaiser window
        # (Copied implementation from afrancl/BinauralLocalizationCNN)
        t = np.arange(-window_length / 2, int(window_length / 2))
        filt = np.sinc(t * (up / down)) * (up / down)
        window = scipy.signal.windows.kaiser(window_length, beta=beta)
        return window * filt
    max_rate = np.max([up, down])
    sfact = np.sqrt(1 + (beta / np.pi) ** 2)
    # Generate first filter attempt (6dB attenuation at f_c).
    filt = scipy.signal.fir_filter_design.firwin(
        window_length,
        1/max_rate,
        window=('kaiser', beta))
    # If nnshift is set to False, simply return the first filter
    if not nnshift:
        return filt
    # Compute frequency response of the first filter
    N_FFT = 2 ** 19
    NBINS = N_FFT / 2 + 1
    paddedfilt = np.zeros(N_FFT)
    paddedfilt[:window_length] = filt
    ffilt = np.fft.rfft(paddedfilt)
    # Find the minimum between f_c and f_c+sqrt(1+(beta/pi)^2)/window_length
    bot = int(np.floor(NBINS / max_rate))
    top = int(np.ceil(NBINS * (1 / max_rate + 2 * sfact / window_length)))
    firstnull = (np.argmin(np.abs(ffilt[bot:top])) + bot) / NBINS
    # Generate a shifted filter with the cutoff on the first null
    shifted_filt = scipy.signal.fir_filter_design.firwin(
        window_length,
        -firstnull+2/max_rate,
        window=('kaiser', beta))
    return shifted_filt


def tfnnresample(tensor_input,
                 sr_input,
                 sr_output,
                 kwargs_nnresample_poly_filter={},
                 verbose=True):
    """
    Tensorflow function for resampling time-domain signals.
    Null-on-Nyquist anti-aliasing lowpass filter is applied.
    
    Args
    ----
    tensor_input (tensor): input tensor to be resampled along time dimension (expects shape
        [batch, time], [batch, freq, time], or [batch, freq, time, 1])
    sr_input (int): input sampling rate in Hz
    sr_output (int): output sampling rate in Hz
    kwargs_nnresample_poly_filter (dict): keyword arguments for nnresample_poly_filter,
        which can be used to alter cutoff frequency of anti-aliasing lowpass filter
        (defaults to (1/2) * sr_output)
    verbose (bool): if True, function will print optional information
    
    Returns
    -------
    tensor_input_resampled (tensor): resampled tensor with shape matched to tensor_input
    """
    # Expand dimensions of input tensor to [batch, freq, time, channels] for 2d conv operation
    if len(tensor_input.shape) == 2:
        if verbose:
            print('[tfnnresample] interpreting `tensor_input.shape` as [batch, time]')
        tensor_input_expanded = tensor_input[:, tf.newaxis, :, tf.newaxis]
    elif len(tensor_input.shape) == 3:
        if verbose:
            print('[tfnnresample] interpreting `tensor_input.shape` as [batch, freq, time]')
        tensor_input_expanded = tensor_input[:, :, :, tf.newaxis]
    else:
        if verbose:
            print('[tfnnresample] interpreting `tensor_input.shape` as [batch, freq, time, channels]')
        tensor_input_expanded = tensor_input
    msg = "dimensions of `tensor_input` must support re-shaping to [batch, freq, time, channels]"
    assert (len(tensor_input_expanded.shape) == 4), msg
    # Compute upsample and downsample factors
    greatest_common_divisor = np.gcd(int(sr_output), int(sr_input))
    up = int(sr_output) // greatest_common_divisor
    down = int(sr_input) // greatest_common_divisor
    # First upsample by a factor of `up` by adding `up-1` zeros between each sample in original signal
    nzeros = up - 1
    if nzeros > 0:
        paddings = [[0,0],[0,0],[0,1],[0,0]] # This will add a zero at the end of the time dimension
        tensor_input_padded = tf.pad(
            tensor_input_expanded,
            paddings,
            mode='CONSTANT',
            constant_values=0)
        indices = []
        for idx in range(tensor_input_expanded.shape[2]):
            indices.append(idx)
            indices.extend([tensor_input_expanded.shape[2]] * nzeros) # This will insert nzeros zeros between each sample
        tensor_input_upsampled = tf.gather(tensor_input_padded, indices, axis=2)
    else:
        tensor_input_upsampled = tensor_input_expanded
    # Next construct lowpass anti-aliasing filter (kwargs_nnresample_poly_filter will override up/down)
    kwargs_nnresample_poly_filter = dict(kwargs_nnresample_poly_filter) # prevents modifying in-place
    if kwargs_nnresample_poly_filter.get('up', None) is None:
        kwargs_nnresample_poly_filter['up'] = up
    elif verbose:
        print('[tfnnresample] using up={} rather than up={} for nnresample_poly_filter'.format(
            kwargs_nnresample_poly_filter['up'], up))
    if kwargs_nnresample_poly_filter.get('down', None) is None:
        kwargs_nnresample_poly_filter['down'] = down
    elif verbose:
        print('[tfnnresample] using down={} rather than down={} for nnresample_poly_filter'.format(
            kwargs_nnresample_poly_filter['down'], down))
    if kwargs_nnresample_poly_filter.get('window_length', None) is None:
        kwargs_nnresample_poly_filter['window_length'] = int(tensor_input_upsampled.shape[2])
        if verbose:
            print('[tfnnresample] using window_length={} for nnresample_poly_filter'.format(
                kwargs_nnresample_poly_filter['window_length']))
    if verbose:
        print('[tfnnresample] using cutoff frequency near {} Hz for anti-aliasing lowpass filter'.format(
            (kwargs_nnresample_poly_filter['up']/kwargs_nnresample_poly_filter['down']) * (sr_input/2)))
    aa_filter_ir = nnresample_poly_filter(**kwargs_nnresample_poly_filter)
    aa_filter_ir_tensor = tf.constant(aa_filter_ir, dtype=tensor_input.dtype)
    aa_filter_ir_tensor = aa_filter_ir_tensor[tf.newaxis, :, tf.newaxis, tf.newaxis]
    tensor_eye = tf.eye(
        num_rows=list(tensor_input_upsampled.shape)[-1],
        num_columns=None,
        batch_shape=[1, 1],
        dtype=tensor_input_upsampled.dtype,
        name=None)
    # Apply the lowpass filter and downsample in one step via strided convolution
    tensor_input_resampled = tf.nn.convolution(
        tensor_input_upsampled,
        aa_filter_ir_tensor * tensor_eye,
        strides=[1, 1, down, 1],
        padding='SAME',
        data_format='NHWC')
    # Reshape resampled output tensor to match dimensions of input tensor
    if len(tensor_input.shape) == 2:
        tensor_input_resampled = tf.squeeze(tensor_input_resampled, axis=[1, -1])
    elif len(tensor_input.shape) == 3:
        tensor_input_resampled = tf.squeeze(tensor_input_resampled, axis=[-1])
    return tensor_input_resampled


def tf_scipy_signal_resample_poly(x, up, down, axis=-1, **kwargs):
    """
    Tensorflow wrapper for scipy.signal.resample_poly to convert
    audio sampling rate along specified axis (NOTE: gradients do
    not compute through tf.numpy_function).
    """
    shape = list(x.shape)
    shape[axis] = int(shape[axis] * up / down)
    numpy_func = lambda _: scipy.signal.resample_poly(_, up=up, down=down, axis=axis, **kwargs).astype(np.float32)
    x = tf.numpy_function(numpy_func, [x], tf.float32)
    x = tf.ensure_shape(tf.cast(x, tf.float32), shape)
    return x
