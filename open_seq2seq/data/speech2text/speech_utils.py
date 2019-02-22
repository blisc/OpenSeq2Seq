# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import math
import os

import h5py
import librosa
import numpy as np
import python_speech_features as psf
import resampy as rs
import scipy.io.wavfile as wave


class PreprocessOnTheFlyException(Exception):
  """ Exception that is thrown to not load preprocessed features from disk;
  recompute on-the-fly.
  This saves disk space (if you're experimenting with data input
  formats/preprocessing) but can be slower.
  The slowdown is especially apparent for small, fast NNs."""
  pass


class RegenerateCacheException(Exception):
  """ Exception that is thrown to force recomputation of (preprocessed) features
  """
  pass


def load_features(path, data_format):
  """ Function to load (preprocessed) features from disk

  Args:
      :param path:    the path where the features are stored
      :param data_format:  the format in which the features are stored
      :return:        tuple of (features, duration)
      """
  if data_format == 'hdf5':
    with h5py.File(path + '.hdf5', "r") as hf5_file:
      features = hf5_file["features"][:]
      duration = hf5_file["features"].attrs["duration"]
  elif data_format == 'npy':
    features, duration = np.load(path + '.npy')
  elif data_format == 'npz':
    data = np.load(path + '.npz')
    features = data['features']
    duration = data['duration']
  else:
    raise ValueError("Invalid data format for caching: ", data_format, "!\n",
                     "options: hdf5, npy, npz")
  return features, duration


def save_features(features, duration, path, data_format, verbose=False):
  """ Function to save (preprocessed) features to disk

  Args:
      :param features:            features
      :param duration:            metadata: duration in seconds of audio file
      :param path:                path to store the data
      :param data_format:              format to store the data in ('npy',
      'npz',
      'hdf5')
  """
  if verbose: print("Saving to: ", path)

  if data_format == 'hdf5':
    with h5py.File(path + '.hdf5', "w") as hf5_file:
      dset = hf5_file.create_dataset("features", data=features)
      dset.attrs["duration"] = duration
  elif data_format == 'npy':
    np.save(path + '.npy', [features, duration])
  elif data_format == 'npz':
    np.savez(path + '.npz', features=features, duration=duration)
  else:
    raise ValueError("Invalid data format for caching: ", data_format, "!\n",
                     "options: hdf5, npy, npz")


def get_preprocessed_data_path(filename, params):
  """ Function to convert the audio path into the path to the preprocessed
  version of this audio
  Args:
      :param filename:    WAVE filename
      :param params:      dictionary containing preprocessing parameters
      :return:            path to new file (without extension). The path is
      generated from the relevant preprocessing parameters.
  """
  if isinstance(filename, bytes):  # convert binary string to normal string
      filename = filename.decode('ascii')

  filename = os.path.realpath(filename)  # decode symbolic links

  ## filter relevant parameters # TODO is there a cleaner way of doing this?
  # print(list(params.keys()))
  ignored_params = ["cache_features", "cache_format", "cache_regenerate",
                    "vocab_file", "dataset_files", "shuffle", "batch_size",
                    "max_duration",
                    "mode", "interactive", "autoregressive", "char2idx",
                    "tgt_vocab_size", "idx2char", "dtype"]

  def fix_kv(text):
    """ Helper function to shorten length of filenames to get around
    filesystem path length limitations"""
    text = str(text)
    text = text.replace("time_stretch_ratio", "tsr") \
      .replace("noise_level_min", "nlmin", ) \
      .replace("noise_level_max", "nlmax") \
      .replace("add_derivatives", "d") \
      .replace("add_second_derivatives", "dd")
    return text

  # generate the identifier by simply concatenating preprocessing key-value
  # pairs as strings.
  preprocess_id = "-".join(
      [fix_kv(k) + "_" + fix_kv(v) for k, v in params.items() if
       k not in ignored_params])

  preprocessed_dir = os.path.dirname(filename).replace("wav",
                                                       "preprocessed-" +
                                                       preprocess_id)
  preprocessed_path = os.path.join(preprocessed_dir,
                                   os.path.basename(filename).replace(".wav",
                                                                      ""))

  # create dir if it doesn't exist yet
  if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

  return preprocessed_path


def get_speech_features_from_file(filename,
                                  num_features,
                                  pad_to=8,
                                  features_type='spectrogram',
                                  window_size=20e-3,
                                  window_stride=10e-3,
                                  augmentation=None,
                                  cache_features=False,
                                  cache_format="hdf5",
                                  cache_regenerate=False,
                                  params={},
                                  mel_basis=None):
  """Function to get a numpy array of features, from an audio file.
      if params['cache_features']==True, try load preprocessed data from
      disk, or store after preprocesseng.
      else, perform preprocessing on-the-fly.

Args:
  filename (string): WAVE filename.
  num_features (int): number of speech features in frequency domain.
  features_type (string): 'mfcc' or 'spectrogram'.
  window_size (float): size of analysis window in milli-seconds.
  window_stride (float): stride of analysis window in milli-seconds.
  augmentation (dict, optional): None or dictionary of augmentation parameters.
      If not None, has to have 'time_stretch_ratio',
      'noise_level_min', 'noise_level_max' fields, e.g.::
        augmentation={
          'time_stretch_ratio': 0.2,
          'noise_level_min': -90,
          'noise_level_max': -46,
        }
Returns:
  np.array: np.array of audio features with shape=[num_time_steps,
  num_features].
"""
  try:
    if not cache_features:
      raise PreprocessOnTheFlyException(
          "on-the-fly preprocessing enforced with 'cache_features'==True")

    if cache_regenerate:
      raise RegenerateCacheException("regenerating cache...")

    preprocessed_data_path = get_preprocessed_data_path(filename, params)
    features, duration = load_features(preprocessed_data_path,
                                       data_format=cache_format)

  except PreprocessOnTheFlyException:
    if params.get("librosa", False):
      signal, sample_freq = librosa.core.load(filename, sr=None)
      features, duration = get_speech_features_librosa(
          signal, sample_freq, num_features, pad_to, features_type,
          window_size, window_stride, augmentation, mel_basis, params.get("normalize", False)
      )
    else:
      sample_freq, signal = wave.read(filename)
      features, duration = get_speech_features(
          signal, sample_freq, num_features, pad_to, features_type,
          window_size, window_stride, augmentation, mel_basis
      )

  except (OSError, FileNotFoundError, RegenerateCacheException):
    sample_freq, signal = wave.read(filename)
    if params.get("librosa", False):
      features, duration = get_speech_features_librosa(
          signal, sample_freq, num_features, pad_to, features_type,
          window_size, window_stride, augmentation, mel_basis
      )
    else:
      features, duration = get_speech_features(
          signal, sample_freq, num_features, pad_to, features_type,
          window_size, window_stride, augmentation, mel_basis
      )
    preprocessed_data_path = get_preprocessed_data_path(filename, params)
    save_features(features, duration, preprocessed_data_path,
                  data_format=cache_format)

  return features, duration


def normalize_signal(signal):
  """
  Normalize float32 signal to [-1, 1] range
  """
  return signal / np.max(np.abs(signal))


def augment_audio_signal(signal, sample_freq, augmentation):
  """Function that performs audio signal augmentation.

  Args:
    signal (np.array): np.array containing raw audio signal.
    sample_freq (float): frames per second.
    augmentation (dict): dictionary of augmentation parameters. See
        :func:`get_speech_features_from_file` for specification and example.
  Returns:
    np.array: np.array with augmented audio signal.
  """
  signal_float = normalize_signal(signal.astype(np.float32))

  if augmentation['time_stretch_ratio'] > 0:
    # time stretch (might be slow)
    stretch_amount = 1.0 + (2.0 * np.random.rand() - 1.0) * \
                     augmentation['time_stretch_ratio']
    signal_float = rs.resample(
        signal_float,
        sample_freq,
        int(sample_freq * stretch_amount),
        filter='kaiser_fast',
    )

  # noise
  noise_level_db = np.random.randint(low=augmentation['noise_level_min'],
                                     high=augmentation['noise_level_max'])
  signal_float += np.random.randn(signal_float.shape[0]) * \
                  10.0 ** (noise_level_db / 20.0)

  return (normalize_signal(signal_float) * 32767.0).astype(np.int16)


def get_speech_features(signal, sample_freq, num_features, pad_to=8,
                        features_type='spectrogram',
                        window_size=20e-3,
                        window_stride=10e-3,
                        augmentation=None):
  """Function to convert raw audio signal to numpy array of features.

  Args:
    signal (np.array): np.array containing raw audio signal.
    sample_freq (float): frames per second.
    num_features (int): number of speech features in frequency domain.
    pad_to (int): if specified, the length will be padded to become divisible
        by ``pad_to`` parameter.
    features_type (string): 'mfcc' or 'spectrogram'.
    window_size (float): size of analysis window in milli-seconds.
    window_stride (float): stride of analysis window in milli-seconds.
    augmentation (dict, optional): dictionary of augmentation parameters. See
        :func:`get_speech_features_from_file` for specification and example.

  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
    audio_duration (float): duration of the signal in seconds
  """
  if augmentation:
    if 'time_stretch_ratio' not in augmentation:
      raise ValueError('time_stretch_ratio has to be included in augmentation '
                       'when augmentation it is not None')
    if 'noise_level_min' not in augmentation:
      raise ValueError('noise_level_min has to be included in augmentation '
                       'when augmentation it is not None')
    if 'noise_level_max' not in augmentation:
      raise ValueError('noise_level_max has to be included in augmentation '
                       'when augmentation it is not None')
    signal = augment_audio_signal(signal, sample_freq, augmentation)
  else:
    signal = (normalize_signal(signal.astype(np.float32)) * 32767.0).astype(
        np.int16)

  audio_duration = len(signal) * 1.0 / sample_freq

  n_window_size = int(sample_freq * window_size)
  n_window_stride = int(sample_freq * window_stride)

  # making sure length of the audio is divisible by 8 (fp16 optimization)
  length = 1 + int(math.ceil(
      (1.0 * signal.shape[0] - n_window_size) / n_window_stride
  ))
  if pad_to > 0:
    if length % pad_to != 0:
      pad_size = (pad_to - length % pad_to) * n_window_stride
      signal = np.pad(signal, (0, pad_size), mode='constant')

  if features_type == 'spectrogram':
    frames = psf.sigproc.framesig(sig=signal,
                                  frame_len=n_window_size,
                                  frame_step=n_window_stride,
                                  winfunc=np.hanning)

    # features = np.log1p(psf.sigproc.powspec(frames, NFFT=N_window_size))
    features = psf.sigproc.logpowspec(frames, NFFT=n_window_size)
    assert num_features <= n_window_size // 2 + 1, \
      "num_features for spectrogram should be <= (sample_freq * window_size // 2 + 1)"

    # cut high frequency part
    features = features[:, :num_features]

  elif features_type == 'mfcc':
    features = psf.mfcc(signal=signal,
                        samplerate=sample_freq,
                        winlen=window_size,
                        winstep=window_stride,
                        numcep=num_features,
                        nfilt=2 * num_features,
                        nfft=512,
                        lowfreq=0, highfreq=None,
                        preemph=0.97,
                        ceplifter=2 * num_features,
                        appendEnergy=False)

  elif features_type == 'logfbank':
    features = psf.logfbank(signal=signal,
                            samplerate=sample_freq,
                            winlen=window_size,
                            winstep=window_stride,
                            nfilt=num_features,
                            nfft=512,
                            lowfreq=0, highfreq=sample_freq / 2,
                            preemph=0.97)

  else:
    raise ValueError('Unknown features type: {}'.format(features_type))

  if pad_to > 0:
    assert features.shape[0] % pad_to == 0
  mean = np.mean(features)
  std_dev = np.std(features)
  features = (features - mean) / std_dev
  return features, audio_duration

def get_speech_features_librosa(signal, sample_freq, num_features, pad_to=8,
                        features_type='spectrogram',
                        window_size=20e-3,
                        window_stride=10e-3,
                        augmentation=None,
                        mel_basis=None,
                        normalize=False):
  """Function to convert raw audio signal to numpy array of features.

  Args:
    signal (np.array): np.array containing raw audio signal.
    sample_freq (float): frames per second.
    num_features (int): number of speech features in frequency domain.
    pad_to (int): if specified, the length will be padded to become divisible
        by ``pad_to`` parameter.
    features_type (string): 'mfcc' or 'spectrogram'.
    window_size (float): size of analysis window in milli-seconds.
    window_stride (float): stride of analysis window in milli-seconds.
    augmentation (dict, optional): dictionary of augmentation parameters. See
        :func:`get_speech_features_from_file` for specification and example.

  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
    audio_duration (float): duration of the signal in seconds
  """
  audio_duration = len(signal) * 1.0 / sample_freq

  n_window_size = int(sample_freq * window_size)
  n_window_stride = int(sample_freq * window_stride)

  # # making sure length of the audio is divisible by 8 (fp16 optimization)
  # length = 1 + int(math.ceil(
  #     (1.0 * signal.shape[0] - n_window_size) / n_window_stride
  # ))
  # if pad_to > 0:
  #   if length % pad_to != 0:
  #     pad_size = (pad_to - length % pad_to) * n_window_stride
  #     signal = np.pad(signal, (0, pad_size), mode='constant')

  mag, _ = librosa.magphase(librosa.stft(y=signal, n_fft=512, hop_length=n_window_stride, win_length=n_window_size), power=1)
  # frames = psf.sigproc.framesig(sig=signal,
  #                                 frame_len=n_window_size,
  #                                 frame_step=n_window_stride,
  #                                 winfunc=np.hanning)

  # features = np.log1p(psf.sigproc.powspec(frames, NFFT=N_window_size))
  if features_type == 'spectrogram':
    # features = psf.sigproc.logpowspec(frames, NFFT=n_window_size)
    # assert num_features <= n_window_size // 2 + 1, \
    #   "num_features for spectrogram should be <= (sample_freq * window_size // 2 + 1)"

    # # cut high frequency part
    # features = features[:, :num_features]
    features = np.log(np.clip(mag, a_min=1e-5, a_max=None)).T
    assert num_features <= n_window_size // 2 + 1, \
        "num_features for spectrogram should be <= (fs * window_size // 2 + 1)"

    # cut high frequency part
    features = features[:, :num_features]
    pad_value = 1e-5

  # elif features_type == 'mfcc':
  #   features = psf.mfcc(signal=signal,
  #                       samplerate=sample_freq,
  #                       winlen=window_size,
  #                       winstep=window_stride,
  #                       numcep=num_features,
  #                       nfilt=2 * num_features,
  #                       nfft=512,
  #                       lowfreq=0, highfreq=None,
  #                       preemph=0.97,
  #                       ceplifter=2 * num_features,
  #                       appendEnergy=False)

  elif features_type == 'logfbank':
    # features = psf.logfbank(signal=signal,
    #                         samplerate=sample_freq,
    #                         winlen=window_size,
    #                         winstep=window_stride,
    #                         nfilt=num_features,
    #                         nfft=512,
    #                         lowfreq=0, highfreq=sample_freq / 2,
    #                         preemph=0.97)
    features = np.dot(mel_basis, mag)
    features = np.log(np.clip(features, a_min=1e-2, a_max=None)).T
    pad_value = 1e-2

  else:
    raise ValueError('Unknown features type: {}'.format(features_type))

  if pad_to > 0:
    num_pad = pad_to - ((len(features) + 1) % pad_to) + 1
    features = np.pad(
        features,
        # ((8, num_pad), (0, 0)),
        ((0, num_pad), (0, 0)),
        "constant",
        constant_values=pad_value
    )
    assert features.shape[0] % pad_to == 0

  if normalize:
    mean = np.mean(features)
    std_dev = np.std(features)
    features = (features - mean) / std_dev
  return features, audio_duration
