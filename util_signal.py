import os
import sys
import pdb
import numpy as np
import tensorflow as tf
import scipy.signal

from util_stimuli import freq2erb, erb2freq, erbspace


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


def tf_hilbert(x, axis=-1, dtype=tf.dtypes.complex64):
    """
    Compute the analytic signal, using the Hilbert transform.
    
    Args
    ----
    x (array_like): real-valued signal data (x.shape[axis] must be even)
    axis (int): axis along which to do the transformation
    dtype (tf.dtypes): complex datatype for tensorflow FFT
    
    Returns
    -------
    xa (ndarray): Analytic signal of `x`, of each 1-D array along axis=-1
    """
    if not axis == -1:
        x = tf.experimental.numpy.swapaxes(x, axis, -1)
    Xf = tf.signal.fft(tf.cast(x, dtype))
    shape = [1 for _ in range(len(x.shape))]
    shape[-1] = tf.shape(Xf)[-1]
    h = tf.concat([
        tf.ones(1, dtype=Xf.dtype),
        2 * tf.ones([tf.shape(Xf)[-1] // 2 - 1], dtype=Xf.dtype),
        tf.ones(1, dtype=Xf.dtype),
        tf.zeros([tf.shape(Xf)[-1] // 2 - 1], dtype=Xf.dtype),
    ], axis=0)
    h = tf.reshape(h, shape)
    xa = tf.signal.ifft(Xf * h)
    if not axis == -1:
        xa = tf.experimental.numpy.swapaxes(xa, -1, axis)
    return xa


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


def half_cosine_filterbank(tensor_input,
                           sr,
                           num_cf=50,
                           min_lo=20.0,
                           max_hi=10e3,
                           return_io_function=True):
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
    filts, freqs = make_half_cosine_filters(
        tensor_input.shape[-1],
        sr,
        list_lo=list_lo,
        list_hi=list_hi)
    rfft_filts = tf.constant(filts[tf.newaxis, :, :], dtype=tf.complex64)
    
    def filterbank_io_function(x):
        # Prepare input signal and apply roex filterbank in frequency domain
        if len(x.shape) == 1:
            x = x[tf.newaxis, :]
        elif len(x.shape) > 2:
            raise ValueError("Input dimensions should be: [batch, time]")
        rfft_x = tf.signal.rfft(tf.cast(x, tf.float32))[:, tf.newaxis, :]
        rfft_y = tf.math.multiply(rfft_x, rfft_filts)
        y = tf.signal.irfft(rfft_y)
        return y
    
    if return_io_function:
        return filterbank_io_function
    else:
        return filterbank_io_function(tensor_input)


def fir_lowpass_filter(sr_input,
                       sr_output,
                       numtaps=None,
                       fir_dur=None,
                       cutoff=None,
                       order=None,
                       ihc_filter=False,
                       window=('kaiser', 5.0),
                       verbose=False):
    """
    Build a finite impulse response lowpass filter (e.g., for anti-aliasing).
    
    Args
    ----
    sr_input (int): input sampling rate (Hz)
    sr_output (int): output sampling rate (Hz)
    numtaps (int): length of FIR in samples at filter sampling rate
    fir_dur (float): duration of finite impulse response in seconds
    cutoff (int): lowpass filter cutoff frequency in Hz
    order (int): lowpass filter order (if using `ihc_filter`)
    ihc_filter (bool): approximate FIR from bez2018model IHC lowpass filter
    window (tuple): argument for `scipy.signal.windows.get_window`
    
    Returns
    -------
    filt (np.ndarray): finite impulse response of lowpass filter
    sr_filt (int): sampling rate of lowpass filter impulse response
    """
    none_args = [_ for _ in [numtaps, fir_dur] if _ is None]
    assert len(none_args) == 1, "Specify exactly one of [numtaps, fir_dur]"
    if sr_output is None:
        sr_output = sr_input
    greatest_common_divisor = np.gcd(int(sr_input), int(sr_output))
    down = int(sr_input) // greatest_common_divisor
    up = int(sr_output) // greatest_common_divisor
    sr_filt = sr_input * up
    if ihc_filter:
        assert cutoff is not None, "cutoff must be specified for ihc lowpass filter"
        assert order is not None, "order must be specified for ihc lowpass filter"
    if cutoff is None:
        cutoff = sr_output / 2
    if fir_dur is None:
        fir_dur = int(2 * (numtaps // 2)) / sr_filt
    else:
        numtaps = int(2 * (fir_dur * sr_filt // 2)) + 1 # Ensure numtaps from fir_dur is odd
    assert cutoff <= sr_output / 2, "cutoff may not exceed Nyquist"
    if ihc_filter:
        # FIR approximation of IHC lowpass filter from bez2018model/model_IHC_BEZ2018.c
        impulse = np.zeros(numtaps)
        filt = np.zeros_like(impulse)
        impulse[0] = 1
        ihc = np.zeros(order + 1)
        ihcl = np.zeros(order + 1)
        c1LP = (sr_filt - 2 * np.pi * cutoff) / (sr_filt + 2 * np.pi * cutoff)
        c2LP = (np.pi * cutoff) / (sr_filt + 2 * np.pi * cutoff)
        for n in range(len(impulse)):
            ihc[0] = impulse[n]
            for i in range(order):
                ihc[i + 1] = (c1LP * ihcl[i + 1]) + c2LP * (ihc[i] + ihcl[i])
            ihcl = ihc
            filt[n] = ihc[order]
        filt = filt * scipy.signal.windows.hann(len(filt))
        filt = filt / filt.sum()
    else:
        filt = scipy.signal.firwin(
            numtaps=numtaps,
            cutoff=cutoff,
            width=None,
            window=tuple(window),
            pass_zero=True,
            scale=True,
            fs=sr_filt)
    if verbose:
        print("[fir_lowpass_filter] sr_filt = {} Hz".format(sr_filt))
        print("[fir_lowpass_filter] numtaps = {} samples".format(numtaps))
        print("[fir_lowpass_filter] fir_dur = {} seconds".format(fir_dur))
        print("[fir_lowpass_filter] cutoff = {} Hz".format(cutoff))
        if ihc_filter:
            print("[fir_lowpass_filter] order = {} (bez2018model IHC filter)".format(order))
        else:
            print("[fir_lowpass_filter] window = {}".format(tuple(window)))
    return filt, sr_filt


def tf_fir_resample(tensor_input,
                    sr_input,
                    sr_output,
                    kwargs_fir_lowpass_filter={},
                    verbose=True,
                    return_io_function=True):
    """
    Tensorflow function for resampling time-domain signals with an FIR lowpass filter.
    
    Args
    ----
    tensor_input (tensor): input tensor to be resampled along time dimension (expects shape
        [batch, time], [batch, freq, time], or [batch, freq, time, 1])
    sr_input (int): input sampling rate in Hz
    sr_output (int): output sampling rate in Hz
    kwargs_fir_lowpass_filter (dict): keyword arguments for fir_lowpass_filter,
        which can be used to alter cutoff frequency of anti-aliasing lowpass filter
    verbose (bool): if True, function will print optional information
    return_io_function (bool): if True, function will return a wrapper function that
        applies resampling operation. Otherwise, function will return resampled tensor
    
    Returns
    -------
    resample_io_function (function): wrapper function maps tensors to resampled tensors
    < OR >
    tensor_input_resampled (tensor): resampled tensor with shape matched to tensor_input
    """
    # Expand dimensions of input tensor to [batch, freq, time, channels] for 2d conv operation
    if len(tensor_input.shape) == 2:
        f = 1
        t = tensor_input.shape[1]
        c = 1
        expand_io_function = lambda x: x[:, tf.newaxis, :, tf.newaxis]
        if verbose:
            print(f'[tf_fir_resample] interpreted `tensor_input.shape` as [batch, time={t}]')
    elif len(tensor_input.shape) == 3:
        f = tensor_input.shape[1]
        t = tensor_input.shape[2]
        c = 1
        expand_io_function = lambda x: x[:, :, :, tf.newaxis]
        if verbose:
            print(f'[tf_fir_resample] interpreted `tensor_input.shape` as [batch, freq={f}, time={t}]')
    else:
        msg = "dimensions of `tensor_input` must support re-shaping to [batch, freq, time, channels]"
        assert (len(tensor_input.shape) == 4), msg
        f = tensor_input.shape[1]
        t = tensor_input.shape[2]
        c = tensor_input.shape[3]
        expand_io_function = lambda x: x
        if verbose:
            print(f'[tf_fir_resample] interpreted `tensor_input.shape` as [batch, freq={f}, time={t}, channels={c}]')
    # Compute upsample and downsample factors
    greatest_common_divisor = np.gcd(int(sr_output), int(sr_input))
    up = int(sr_output) // greatest_common_divisor
    down = int(sr_input) // greatest_common_divisor
    # First upsample by a factor of `up` by adding `up-1` zeros between each sample in original signal
    nzeros = up - 1
    if nzeros > 0:
        paddings = [(0,0), (0,0), (0,1), (0,0)]
        indices = []
        for _ in range(t):
            indices.append(_)
            indices.extend([t] * nzeros)
        def upsample_io_function(x):
            # Add one zero at the end of the time dimension
            x_padded = tf.pad(
                x,
                paddings=paddings,
                mode='CONSTANT',
                constant_values=0)
            # Insert up - 1 zeros between each sample
            return tf.gather(x_padded, indices, axis=2)
    else:
        upsample_io_function = lambda _: _
    # Next construct anti-aliasing lowpass filter
    kwargs_fir_lowpass_filter = dict(kwargs_fir_lowpass_filter) # prevents modifying in-place
    if kwargs_fir_lowpass_filter.get('cutoff', None) is None:
        kwargs_fir_lowpass_filter['cutoff'] = sr_output / 2
    if verbose:
        print('[tf_fir_resample] `kwargs_fir_lowpass_filter`: {}'.format(kwargs_fir_lowpass_filter))
    filt, sr_filt = fir_lowpass_filter(sr_input, sr_output, **kwargs_fir_lowpass_filter, verbose=verbose)
    filt = filt * up # Re-scale filter to offset attenuation from upsampling
    if kwargs_fir_lowpass_filter.get('ihc_filter', False):
        filt = filt[::-1] # IHC lowpass filter needs to be time-inverted and padded
        n_pad = len(filt) # More efficient to add padding to signal and use 'VALID' convolution
        pad_io_function = lambda _: tf.pad(_, [(0,0), (0,0), (n_pad - 1,0), (0,0)])
        padding = 'VALID'
    else:
        pad_io_function = tf.identity
        padding = 'SAME'
    filt = tf.constant(filt, dtype=tensor_input.dtype)[tf.newaxis, :, tf.newaxis, tf.newaxis]
    tensor_kernel_lowpass_filter = filt * tf.eye(
        num_rows=c,
        num_columns=None,
        batch_shape=[1, 1],
        dtype=tensor_input.dtype,
        name=None)
    
    def downsample_io_function(x):
        # Apply lowpass filter and downsample in one step via strided convolution
        return tf.nn.convolution(
            pad_io_function(x),
            tensor_kernel_lowpass_filter,
            strides=[1, 1, down, 1],
            padding=padding,
            data_format='NHWC')
    
    # Reshape resampled output tensor to match dimensions of input tensor
    if len(tensor_input.shape) == 2:
        squeeze_io_function = lambda _: tf.squeeze(_, axis=[1, -1])
    elif len(tensor_input.shape) == 3:
        squeeze_io_function = lambda _: tf.squeeze(_, axis=[-1])
    else:
        squeeze_io_function = lambda _: _
    
    def resample_io_function(x):
        # Executes full resampling operation (expand dims -> upsample -> downsample -> squeeze dims)
        return squeeze_io_function(downsample_io_function(upsample_io_function(expand_io_function(x))))
    
    if return_io_function:
        return resample_io_function
    else:
        return resample_io_function(tensor_input)


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


def tf_spatialize_sound(x, brir, return_io_function=True):
    """
    Tensorflow function for convolving a monaural signals (or batch of signals)
    with a binaural room impulse response (with shape [time, 2]) to generate a
    spatialized binaural waveform.
    """
    assert (len(brir.shape) == 2) and (brir.shape[-1] == 2), "expected brir shape [time, 2]"
    tensor_filter = tf.cast(brir[tf.newaxis, ::-1, :, tf.newaxis], x.dtype)
    if len(x.shape) == 1:
        pad_io_function = lambda _: tf.pad(_, [(brir.shape[0] - 1, 0), (0, 0)])
        reshape_io_function = lambda _: _[tf.newaxis, tf.newaxis, :, :]
        slice_io_function = lambda _: _[0, 0]
    elif len(x.shape) == 2:
        pad_io_function = lambda _: tf.pad(_, [(0, 0), (brir.shape[0] - 1, 0), (0, 0)])
        reshape_io_function = lambda _: _[:, tf.newaxis, :, :]
        slice_io_function = lambda _: _[:, 0]
    else:
        raise ValueError("Expected input shape [time] or [batch, time]")
    
    def spatialize_io_function(_):
        tensor_input = reshape_io_function(pad_io_function(tf.stack([_, _], axis=-1)))
        tensor_output = slice_io_function(
            tf.nn.depthwise_conv2d(
                tensor_input,
                tensor_filter,
                strides=[1, 1, 1, 1],
                padding='VALID',
                data_format=None,
                dilations=None,
                name=None))
        return tensor_output
    
    if return_io_function:
        return spatialize_io_function
    else:
        return spatialize_io_function(x)
