import os
import sys
import argparse
import copy
import glob
import json
import pdb
import numpy as np


def channelwise_dropout(example):
    """
    """
    tmp = tf.constant([
            [[[1, 1, 1]]],
            [[[1, 0, 0]]],
            [[[0, 1, 0]]],
            [[[0, 0, 1]]],
            [[[1, 1, 0]]],
            [[[1, 0, 1]]],
            [[[0, 1, 1]]],
        ],
        dtype=example['nervegram_meanrates'].dtype)
    idx = tf.random.uniform(
        shape=(),
        minval=0,
        maxval=tmp.shape[0],
        dtype=tf.dtypes.int32,
        seed=None,
        name=None)
    example['nervegram_meanrates'] = example['nervegram_meanrates'] * tmp[idx]
    return example


def channelwise_manipulation(example, channels=[], mode='shuffle'):
    """
    """
    nervegram = example['nervegram_meanrates']
    if mode == 'shuffle':
        list_sub_nervegram = tf.unstack(
            nervegram,
            nervegram.shape[-1],
            axis=-1)
        for channel in channels:
            tmp = list_sub_nervegram[channel]
            tmp = tf.reshape(
                tf.random.shuffle(
                    tf.reshape(tmp, [-1])),
                tmp.shape)
            list_sub_nervegram[channel] = tmp
        nervegram = tf.stack(list_sub_nervegram, axis=-1)
    elif mode == 'silence':
        mask = np.ones(nervegram.shape[-1])
        mask[channels] = 0
        mask = mask[np.newaxis, np.newaxis, :]
        nervegram = nervegram * mask
    elif mode == 'substitute':
        list_sub_nervegram = tf.unstack(
            nervegram,
            nervegram.shape[-1],
            axis=-1)
        assert len(channels) == len(list_sub_nervegram)
        list_sub_nervegram = [list_sub_nervegram[_] for _ in channels]
        nervegram = tf.stack(list_sub_nervegram, axis=-1)
    else:
        raise ValueError("`channelwise_manipulation` did not recognize mode={}".format(mode))
    example['nervegram_meanrates'] = nervegram
    return example


def flatten_excitation_pattern(example):
    '''
    Helper function flattens excitation pattern by dividing nervegram
    by time-averaged firing rates and multiplying by frequency- and time-
    averaged firing rates. Frequency is axis 0; time is axis 1.
    '''
    nervegram = tf.cast(example['nervegram_meanrates'], tf.float32)
    nervegram_exc = tf.math.reduce_mean(nervegram, axis=1, keepdims=True)
    nervegram = tf.math.multiply(
        tf.math.divide_no_nan(nervegram, nervegram_exc),
        tf.math.reduce_mean(nervegram, axis=[0, 1], keepdims=True))
    example['nervegram_meanrates'] = tf.cast(nervegram, example['nervegram_meanrates'].dtype)
    return example


def resize_freq_axis(nervegram, n_freq=400):
    """
    Linearly interpolate `nervegram` frequency axis to
    have `n_freq` rows. `nervegram` must have shape
    [freq, time, channel] or [batch, freq, time, channel].
    """
    nervegram = tf.image.resize(
        nervegram,
        size=[
            n_freq,
            nervegram.shape[-2]
        ],
        method='bilinear',
        preserve_aspect_ratio=False,
        antialias=False,
        name='resize_freq_axis')
    return nervegram


def prepare_localization_example(example, signal_slice_length=40000):
    """
    Helper function prepares examples loaded from sound localization
    tfrecord datasets by computing a sound location integer label and
    taking a random 1-second excerpt from `nervegram_meanrates`.
    """
    # Francl localization label convention (labels 0-71 correspond to elevation = 0)
    example['label_loc_int'] = tf.cast(
        tf.math.add(
            (example['foreground_elevation'] / 10) * 72,
            (example['foreground_azimuth'] / 5)
        ),
        tf.int64
    )
    if 'nervegram_meanrates' in example:
        # Randomly slice 1-second excerpt of nervegram
        example['nervegram_meanrates'] = util_cochlea.random_slice(
            example['nervegram_meanrates'],
            slice_length=10000,
            axis=1, # Time axis
            buffer=None)
    if 'signal' in example:
        example['signal'] = util_signal.tf_scipy_signal_resample_poly(
            example['signal'],
            up=40000,
            down=44100,
            axis=-2)
        if signal_slice_length is not None:
            example['signal'] = util_cochlea.random_slice(
                example['signal'],
                slice_length=signal_slice_length,
                axis=-2, # Time axis
                buffer=None)
    return example


def prepare_pitchnet_example(example):
    """
    Slice 150ms waveform to 75ms waveform (4800 --> 2400 samples)
    """
    if 'signal' in example:
        if example['signal'].shape[0] >= 2160+2400:
            example['signal'] = example['signal'][2160:2160+2400]
    return example


def prepare_timit_example(example):
    """
    Resample waveform from 16kHz to 20kHz
    """
    if 'signal' in example:
        if example['signal'].shape[-1] == 32000:
            example['signal'] = util_signal.tf_scipy_signal_resample_poly(
                example['signal'],
                up=20000,
                down=16000,
                axis=-1)
    return example


if __name__ == "__main__":
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description='phaselocknet train routine')
    parser.add_argument('-m', '--dir_model', type=str, default=None, help='model directory')
    parser.add_argument('-c', '--config', type=str, default='config.json', help='config file')
    parser.add_argument('-a', '--arch', type=str, default='arch.json', help='arch file')
    parser.add_argument('-t', '--tfrecords_train', type=str, default=None, help='train dataset')
    parser.add_argument('-v', '--tfrecords_valid', type=str, default=None, help='valid dataset')
    parser.add_argument('-e', '--tfrecords_eval', type=str, default=None, help='evaluation dataset')
    parser.add_argument('-efn', '--basename_eval', type=str, default='EVAL.json')
    parser.add_argument('-ebs', '--batch_size_eval', type=int, default=16)
    parser.add_argument('-wpo', '--write_probs_out', type=int, default=0)
    parser.add_argument('-mp', '--mixed_precision', type=int, default=1)
    args = parser.parse_args()
    # Exit evaluation routine if supplied `basename_eval` already exists
    if (args.tfrecords_eval is not None) and (os.path.exists(os.path.join(args.dir_model, args.basename_eval))):
        print('[SKIP] {} already exists!'.format(os.path.join(args.dir_model, args.basename_eval)))
        sys.exit()
    # Import tensorflow after evaluation routine check for speed
    import tensorflow as tf
    import util_tfrecords
    import util_signal
    import util_cochlea
    import util_network
    import util_optimize
    import util_evaluate
    # Load config file
    fn_config = args.config
    if fn_config == os.path.basename(fn_config):
        fn_config = os.path.join(args.dir_model, fn_config)
    with open(fn_config, 'r') as f_config:
        CONFIG = json.load(f_config)
    # Load network architecture file
    fn_arch = args.arch
    if fn_arch == os.path.basename(fn_arch):
        fn_arch = os.path.join(args.dir_model, fn_arch)
    with open(fn_arch, 'r') as f_arch:
        list_layer_dict = json.load(f_arch)
    # Modify `kwargs_dataset_from_tfrecords` to ensure training labels are accessible
    n_classes_dict = CONFIG.get('n_classes_dict', {})
    kwargs_dataset_from_tfrecords = CONFIG.get('kwargs_dataset_from_tfrecords', {})
    kwargs_dataset_from_tfrecords['map_function'] = []
    if ('label_loc_int' in n_classes_dict):
        if 'cochlearn' in args.dir_model.lower():
            kwargs_dataset_from_tfrecords['map_function'].append(
                lambda _: prepare_localization_example(_, signal_slice_length=41000))
        else:
            kwargs_dataset_from_tfrecords['map_function'].append(
                lambda _: prepare_localization_example(_, signal_slice_length=48000))
    if ('f0_label' in n_classes_dict) and ('cochlearn' in args.dir_model.lower()):
        kwargs_dataset_from_tfrecords['map_function'].append(prepare_pitchnet_example)
    n_freq = None
    if 'nfreq' in args.dir_model:
        tmp = args.dir_model[args.dir_model.rfind('nfreq'):].split('_')[0]
        n_freq = int(''.join(_ for _ in tmp if _.isdigit()))
        BATCH_SCALE_FACTOR = 1
        CONFIG['kwargs_optimize']['batch_size'] = CONFIG['kwargs_optimize']['batch_size'] // BATCH_SCALE_FACTOR
        CONFIG['kwargs_optimize']['steps_per_epoch'] = CONFIG['kwargs_optimize']['steps_per_epoch'] * BATCH_SCALE_FACTOR
        CONFIG['kwargs_optimize']['validation_steps'] = CONFIG['kwargs_optimize']['validation_steps'] * BATCH_SCALE_FACTOR
        CONFIG['kwargs_optimize']['basename_ckpt_epoch'] = 'ckpt_{epoch:04d}'
    if ('channelwise_dropout' in args.dir_model) and (args.tfrecords_eval is None):
        kwargs_dataset_from_tfrecords['map_function'].append(channelwise_dropout)
        print('\n\nIncorporating channelwise_dropout during training\n\n')
    # Set parameters for mixed precision model
    if args.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('MIXED_PRECISION: compute dtype = {}; variable dtype = {}'.format(
            policy.compute_dtype, policy.variable_dtype))
        if 'fc_top' in list_layer_dict[-1]['layer_type']:
            list_layer_dict[-1]['args']['dtype'] = 'float32'
    # Define model_io_function
    def model_io_function(x):
        y = x
        if n_freq is not None:
            y = resize_freq_axis(y, n_freq=n_freq)
            print('[model_io_function] resize_freq_axis(nervegram, n_freq={})'.format(n_freq))
        if CONFIG.get('kwargs_cochlea', {}):
            if 'label_loc_int' in n_classes_dict:
                msg = "expected [batch, freq, time, spont, channel=2] or [batch, time, channel=2]"
                assert (len(y.shape) in [3, 5]) and (y.shape[-1] == 2), msg
                y0, _ = util_cochlea.cochlea(y[..., 0], **copy.deepcopy(CONFIG['kwargs_cochlea']))
                y1, _ = util_cochlea.cochlea(y[..., 1], **copy.deepcopy(CONFIG['kwargs_cochlea']))
                y = tf.concat([y0, y1], axis=-1)
            else:
                y, _ = util_cochlea.cochlea(y, **copy.deepcopy(CONFIG['kwargs_cochlea']))
        y, _ = util_network.build_network(y, list_layer_dict, n_classes_dict=n_classes_dict)
        return y
    
    if args.tfrecords_eval is not None:
        # Run evaluation
        assert args.tfrecords_train is None
        assert args.tfrecords_valid is None
        print('[:::::::::::::::::::::]')
        print('[:: MODEL DIRECTORY ::] {}'.format(args.dir_model))
        print('[:: FILENAME CONFIG ::] {}'.format(fn_config))
        print('[:: FILENAME ARCH   ::] {}'.format(fn_arch))
        print('[:: TFRECORDS EVAL  ::] {} ({} files)'.format(
            args.tfrecords_eval, len(glob.glob(args.tfrecords_eval))))
        print('[:::::::::::::::::::::]')
        if '_spont' in args.basename_eval:
            channels = []
            spont_tag = args.basename_eval[args.basename_eval.rfind('spont'):args.basename_eval.rfind('.json')]
            if 'H' in spont_tag:
                channels.append(0)
            if 'M' in spont_tag:
                channels.append(1)
            if 'L' in spont_tag:
                channels.append(2)
            if 'shuffle_spont' in args.basename_eval:
                mode = 'shuffle'
            if 'silence_spont' in args.basename_eval:
                mode = 'silence'
            if 'substitute_spont' in args.basename_eval:
                mode = 'substitute'
            map_function = lambda _: channelwise_manipulation(_, channels=channels, mode=mode)
            kwargs_dataset_from_tfrecords['map_function'].insert(0, map_function)
            print('{} --> set `map_function` as channelwise_manipulation(example, channels={}, mode={})'.format(
                spont_tag,
                channels,
                mode))
        if ('flat_exc' in args.basename_eval):
            kwargs_dataset_from_tfrecords['map_function'].append(flatten_excitation_pattern)
            print('{} --> set `map_function` as flatten_excitation_pattern(example)'.format(args.basename_eval))
        if ('timit' in args.tfrecords_eval):
            kwargs_dataset_from_tfrecords['map_function'].append(prepare_timit_example)
            print('\n\nResampling TIMIT waveforms from 16kHz to 20kHz\n\n')
        util_evaluate.evaluate(
            tfrecords=args.tfrecords_eval,
            dataset=None,
            key_inputs=CONFIG['kwargs_optimize']['key_inputs'],
            key_outputs=CONFIG['kwargs_optimize']['key_outputs'],
            key_activations=[],
            model_io_function=model_io_function,
            kwargs_loss=CONFIG['kwargs_optimize'].get('kwargs_loss', {}),
            kwargs_dataset_from_tfrecords=kwargs_dataset_from_tfrecords,
            batch_size=args.batch_size_eval,
            dir_model=args.dir_model,
            basename_ckpt=CONFIG['kwargs_optimize'].get('basename_ckpt_best', 'ckpt_BEST'),
            basename_eval=args.basename_eval,
            keys_to_ignore=[],
            write_probs_out=args.write_probs_out,
            disp_step=50)
    else:
        # Run optimization
        assert args.tfrecords_eval is None
        print('[:::::::::::::::::::::]')
        print('[:: MODEL DIRECTORY ::] {}'.format(args.dir_model))
        print('[:: FILENAME CONFIG ::] {}'.format(fn_config))
        print('[:: FILENAME ARCH   ::] {}'.format(fn_arch))
        print('[:: TFRECORDS TRAIN ::] {} ({} files)'.format(
            args.tfrecords_train, len(glob.glob(args.tfrecords_train))))
        print('[:: TFRECORDS VALID ::] {} ({} files)'.format(
            args.tfrecords_valid, len(glob.glob(args.tfrecords_valid))))
        print('[:::::::::::::::::::::]')
        list_gpu = tf.config.list_physical_devices(device_type='GPU')
        strategy = tf.distribute.MirroredStrategy()
        print('\n\nAVAILABLE GPU DEVICES: {}'.format([gpu.name for gpu in list_gpu]))
        print('NUMBER OF PARALLEL TOWERS: {}\n\n'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            history = util_optimize.optimize(
                dir_model=args.dir_model,
                tfrecords_train=args.tfrecords_train,
                tfrecords_valid=args.tfrecords_valid,
                dataset_train=None,
                dataset_valid=None,
                model_io_function=model_io_function,
                kwargs_dataset_from_tfrecords=kwargs_dataset_from_tfrecords,
                **CONFIG['kwargs_optimize'])
