# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import os
import six
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd

from six import string_types

from open_seq2seq.data.data_layer import DataLayer
from open_seq2seq.data.utils import load_pre_existing_vocabulary
from .speech_utils import get_speech_features_from_file, get_mel,\
                          inverse_mel, normalize, denormalize
from .text import text_to_sequence, sequence_to_text


class Text2SpeechDataLayer(DataLayer):
  """Text-to-speech data layer class
  """

  @staticmethod
  def get_required_params():
    return dict(
        DataLayer.get_required_params(), **{
            'dataset': ['LJ', 'Librispeech', 'MAILABS-16'],
            'num_audio_features': None,
            'output_type': ['magnitude', 'mel', 'both', 'magnitude_disk', 'mel_disk', 'both_disk'],
            'vocab_file': str,
            'dataset_files': list,
            'feature_normalize': bool,
        }
    )

  @staticmethod
  def get_optional_params():
    return dict(
        DataLayer.get_optional_params(), **{
            'dataset_location': str,
            'pad_to': int,
            'mag_power': int,
            'pad_EOS': bool,
            'pad_value': float,
            'feature_normalize_mean': float,
            'feature_normalize_std': float,
            'trim': bool,
            'data_min': None,
            'duration_min': int,
            'duration_max': int,
            'text_cleaners': bool,
            'mel_type': ['slaney', 'htk'],
            'style_input': [None, 'wav', 'token'],
            "exp_mag": bool
        }
    )

  def __init__(self, params, model, num_workers=None, worker_id=None):
    """Text-to-speech data layer constructor.

    See parent class for arguments description.

    Config parameters:

    * **dataset** (str) --- The dataset to use. Currently 'LJ' for the LJSpeech
      1.1 dataset is supported.
    * **num_audio_features** (int) --- number of audio features to extract.
    * **output_type** (str) --- could be either "magnitude", or "mel".
    * **vocab_file** (str) --- path to vocabulary file.
    * **dataset_files** (list) --- list with paths to all dataset .csv files.
      File is assumed to be separated by "|".
    * **dataset_location** (string) --- string with path to directory where wavs
      are stored.
    * **feature_normalize** (bool) --- whether to normlize the data with a
      preset mean and std
    * **feature_normalize_mean** (bool) --- used for feature normalize.
      Defaults to 0.
    * **feature_normalize_std** (bool) --- used for feature normalize.
      Defaults to 1.
    * **mag_power** (int) --- the power to which the magnitude spectrogram is
      scaled to. Defaults to 1.
      1 for energy spectrogram
      2 for power spectrogram
      Defaults to 2.
    * **pad_EOS** (bool) --- whether to apply EOS tokens to both the text and
      the speech signal. Will pad at least 1 token regardless of pad_to value.
      Defaults to True.
    * **pad_value** (float) --- The value we pad the spectrogram with. Defaults
      to np.log(data_min).
    * **pad_to** (int) --- we pad such that the resulting datapoint is a
      multiple of pad_to.
      Defaults to 8.
    * **trim** (bool) --- Whether to trim silence via librosa or not. Defaults
      to False.
    * **data_min** (float) --- min clip value prior to taking the log. Defaults
      to 1e-5. Please change to 1e-2 if using htk mels.
    * **duration_min** (int) --- Minimum duration in steps for speech signal.
      All signals less than this will be cut from the training set. Defaults to
      0.
    * **duration_max** (int) --- Maximum duration in steps for speech signal.
      All signals greater than this will be cut from the training set. Defaults 
      to 4000.
    * **mel_type** (str): One of ['slaney', 'htk']. Decides which algorithm to
      use to compute mel specs.
      Defaults to htk.

    """
    super(Text2SpeechDataLayer, self).__init__(
        params,
        model,
        num_workers,
        worker_id
    )

    if self.params['dataset'] == 'LJ' or self.params['dataset'] == 'MAILABS-16':
      if self.params.get('dataset_location', None) is None:
        raise ValueError(
            "dataset_location must be specified when using the LJSpeech or",
            "MAILABS datasets"
        )
      names = ['wav_filename', 'raw_transcript', 'transcript']
      sep = '\x7c'
      header = None
    elif self.params["dataset"] == "Librispeech":
      names = None
      sep = ','
      header = 0
      self.params["dataset_location"] = None

    if self.params['dataset'] == 'LJ':
      self._sampling_rate = 22050
      self._n_fft = 1024
    elif (self.params['dataset'] == 'MAILABS-16' or
          self.params['dataset'] == 'Librispeech'):
      self._sampling_rate = 16000
      self._n_fft = 800
      # self._sampling_rate = 22050
      # self._n_fft = 1024

    # Character level vocab
    self.params['char2idx'] = load_pre_existing_vocabulary(
        self.params['vocab_file'],
        min_idx=3,
        read_chars=True,
    )
    # Add the pad, start, and end chars
    self.params['char2idx']['<p>'] = 0
    self.params['char2idx']['<s>'] = 1
    self.params['char2idx']['</s>'] = 2
    self.params['idx2char'] = {i: w for w, i in self.params['char2idx'].items()}
    self.params['src_vocab_size'] = len(self.params['char2idx'])

    if "disk" in self.params["output_type"]:
      self._load_from_disk = True
    else:
      self._load_from_disk = False
    
    n_feats = self.params['num_audio_features']
    if "both" in self.params["output_type"]:
      self._both = True
      if self.params["feature_normalize"]:
        raise ValueError(
            "feature normalize is not currently enabled for both mode"
        )
      if not isinstance(n_feats, dict):
        raise ValueError(
            "num_audio_features must be a dictionary for both mode"
        )
      else:
        if ("mel" not in n_feats and
            "magnitude" not in n_feats):
          raise ValueError(
            "num_audio_features must contain mel and magnitude keys"
          )
        elif (not isinstance(n_feats["mel"], int) or
              not isinstance(n_feats["magnitude"], int)):
            raise ValueError(
                "num_audio_features must be a int"
            )
      n_mels = n_feats['mel']
      data_min = self.params.get("data_min", None)
      if data_min is not None:
        if not isinstance(data_min, dict):
          raise ValueError(
              "data_min must be a dictionary for both mode"
          )
        else:
          if "mel" not in data_min and "magnitude" not in data_min:
            raise ValueError(
              "data_min must contain mel and magnitude keys"
            )
          elif (not isinstance(data_min["mel"], float) or 
                not isinstance(data_min["magnitude"], float)):
            raise ValueError(
                "data_min must be a float"
            )
      self._exp_mag = self.params.get("exp_mag", True)
    else:
      if not isinstance(n_feats, int):
        raise ValueError(
            "num_audio_features must be a float for mel or magnitude mode"
        )
      if not isinstance(self.params.get("data_min",1.0), float):
        raise ValueError(
            "data_min must be a float for mel or magnitude mode"
        )
      self._both = False
      self._exp_mag = False
      n_mels = n_feats

    if "mel" in self.params["output_type"]:
      self._mel = True
    else:
      self._mel = False

    if self._mel or self._both:
      htk = True
      norm = None
      if self.params.get('mel_type', 'htk') == 'slaney':
        htk = False
        norm = 1
      self._mel_basis = librosa.filters.mel(
          sr=self._sampling_rate,
          n_fft=self._n_fft,
          n_mels=n_mels,
          htk=htk,
          norm=norm
      )
    else:
      self._mel_basis = None

    if self.params["interactive"]:
      return

    # Load csv files
    self._files = None
    for csvs in params['dataset_files']:
      files = pd.read_csv(
          csvs,
          encoding='utf-8',
          sep=sep,
          header=header,
          names=names,
          quoting=3
      )
      if self._files is None:
        self._files = files
      else:
        self._files = self._files.append(files)

    if self.params['mode'] != 'infer' or self.params.get("style_input", None) == "wav":
      cols = ['wav_filename', 'transcript']
    else:
      cols = 'transcript'

    all_files = self._files.loc[:, cols].values
    self._files = self.split_data(all_files)

    self._size = self.get_size_in_samples()
    self._dataset = None
    self._iterator = None
    self._input_tensors = None

  def split_data(self, data):
    if self.params['mode'] != 'train' and self._num_workers is not None:
      #Decrease num_eval for dev, since most data is thrown out anyways
      if self.params['mode'] == 'eval':
        start = self._worker_id * self.params['batch_size']
        end = start+self.params['batch_size']
        return data[start:end]
      size = len(data)
      start = size // self._num_workers * self._worker_id
      if self._worker_id == self._num_workers - 1:
        end = size
      else:
        end = size // self._num_workers * (self._worker_id + 1)
      return data[start:end]
    return data

  @property
  def iterator(self):
    return self._iterator

  def build_graph(self):
    """Builds data reading graph."""
    self._dataset = tf.data.Dataset.from_tensor_slices(self._files)
    if self.params['shuffle']:
      self._dataset = self._dataset.shuffle(self._size)
    self._dataset = self._dataset.repeat()

    if self._both:
      num_audio_features = self.params['num_audio_features']['mel'] 
      num_audio_features += self.params['num_audio_features']['magnitude']
    else:
      num_audio_features = self.params['num_audio_features']

    if self.params['mode'] != 'infer' or self.params.get("style_input", None) == "wav":
      self._dataset = self._dataset.map(
          lambda line: tf.py_func(
              self._parse_audio_transcript_element,
              [line],
              [tf.int32, tf.int32, self.params['dtype'], self.params['dtype'],\
               tf.int32, self.params['dtype']],
              stateful=False,
          ),
          num_parallel_calls=8,
      )
      self._dataset = self._dataset.map(
          self.get_mel,
          num_parallel_calls=8,
      )
      if (self.params.get("duration_max", None) or 
          self.params.get("duration_max", None)):
        self._dataset = self._dataset.filter(
            lambda txt, txt_len, spec, stop, spec_len: 
                tf.logical_and(
                    tf.less_equal(spec_len, self.params.get("duration_max", np.inf)),
                    tf.greater_equal(spec_len, self.params.get("duration_min", 0))
                )
        )
      if self._both:
        default_pad_value = 0.
      else:
        default_pad_value = np.log(self.params.get("data_min", 1e-5))
      pad_value = self.params.get("pad_value", default_pad_value)
      if self.params["feature_normalize"]:
        pad_value = self._normalize(pad_value)
      self._dataset = self._dataset.padded_batch(
          self.params['batch_size'],
          padded_shapes=(
              [None], 1, [None, num_audio_features], [None], 1
          ),
          padding_values=(
              0, 0, tf.cast(pad_value, dtype=self.params['dtype']),
              tf.cast(1., dtype=self.params['dtype']), 0
          )
      )
    else:
      self._dataset = self._dataset.map(
          lambda line: tf.py_func(
              self._parse_transcript_element,
              [line],
              [tf.int32, tf.int32],
              stateful=False,
          ),
          num_parallel_calls=8,
      )
      self._dataset = self._dataset.padded_batch(
          self.params['batch_size'], padded_shapes=([None], 1)
      )

    self._iterator = self._dataset.prefetch(tf.contrib.data.AUTOTUNE)\
                                  .make_initializable_iterator()

    if self.params['mode'] != 'infer' or self.params.get("style_input", None) == "wav":
      text, text_length, spec, stop_token_target, spec_length = self._iterator\
                                                                    .get_next()
      # need to explicitly set batch size dimension
      # (it is employed in the model)
      spec.set_shape(
          [self.params['batch_size'], None, num_audio_features]
      )
      stop_token_target.set_shape([self.params['batch_size'], None])
      spec_length = tf.reshape(spec_length, [self.params['batch_size']])
    else:
      text, text_length = self._iterator.get_next()
    text.set_shape([self.params['batch_size'], None])
    text_length = tf.reshape(text_length, [self.params['batch_size']])

    self._input_tensors = {}
    self._input_tensors["source_tensors"] = [text, text_length]
    if self.params.get("style_input", None) == "token":
      tokens = tf.one_hot(indices=list(range(16)), depth=16)
      # random_tokens = tf.nn.softmax(tf.random_uniform([8,8]))
      # tokens = tf.concat([tokens, random_tokens], axis=0)
      tokens = tf.tile(tokens, [2, 1])
      self._input_tensors["source_tensors"].extend([tokens])
    elif self.params.get("style_input", None) == "wav":
      mel_spec = spec
      if self._both:
        mel_spec, _ = tf.split(
            mel_spec,
            [self.params['num_audio_features']['mel'] , self.params['num_audio_features']['magnitude']],
            axis=2
        )
      self._input_tensors["source_tensors"].extend([mel_spec, spec_length])
    if self.params['mode'] != 'infer':
      self._input_tensors['target_tensors'] = [
          spec, stop_token_target, spec_length
      ]

  def get_mel(self, txt, txt_len, spec, stop, spec_len, mel_basis):
    if self._mel and self._mel_basis is not None:
      mag_spec = tf.exp(spec)
      mel_spec = tf.matmul(mag_spec, mel_basis)
      spec = tf.log(tf.clip_by_value(mel_spec, 1e-2, 1000))
    return txt, txt_len, spec, stop, spec_len

  def _parse_audio_transcript_element(self, element):
    """Parses tf.data element from TextLineDataset into audio and text.

    Args:
      element: tf.data element from TextLineDataset.

    Returns:
      tuple: text_input text as `np.array` of ids, text_input length,
      target audio features as `np.array`, stop token targets as `np.array`,
      length of target sequence.

    """
    audio_filename, transcript = element
    transcript = transcript.lower()
    if six.PY2:
      audio_filename = unicode(audio_filename, "utf-8")
      transcript = unicode(transcript, "utf-8")
    elif not isinstance(transcript, string_types):
      audio_filename = str(audio_filename, "utf-8")
      transcript = str(transcript, "utf-8")
    text_input = np.array(
        [self.params['char2idx'][c] for c in transcript]
    )
    pad_to = self.params.get('pad_to', 8)
    if self.params.get("pad_EOS", True):
      # num_pad = pad_to - (len(text_input) % pad_to)
      num_pad = pad_to - ((len(text_input) + 2) % pad_to)
      text_input = np.pad(
          text_input, ((1, 1)),
          "constant",
          constant_values=(
              (self.params['char2idx']["<s>"], self.params['char2idx']["</s>"])
          )
      )
      text_input = np.pad(
          text_input, ((0, num_pad)),
          "constant",
          constant_values=self.params['char2idx']["<p>"]
      )
    if self._load_from_disk:
      if self.params.get("trim", False):
        audio_filename = audio_filename + "_trimmed"
      # audio_filename = audio_filename + "_800"
      if self.params.get('dataset_location', None) is not None:
        if "wavs" in audio_filename:
          file_path = os.path.join(
              self.params['dataset_location'], audio_filename + ".npy"
          )
        else:
          file_path = os.path.join(
              self.params['dataset_location'], "wavs", audio_filename + ".npy"
          )
      else:
        file_path = os.path.join(audio_filename + ".npy")
        # raise ValueError(
        #     "Librispeech does not support this mode yet."
        # )
      mag_spectrogram = np.load(file_path)
      mag_power = self.params.get('mag_power', 2)
      if self._mel or self._both:
        if self._both:
          n_mels = self.params['num_audio_features']['mel']
          data_min = self.params.get("data_min", 1e-2)
          if isinstance(data_min, dict):
            data_min = data_min["mel"]
        else:
          n_mels = self.params['num_audio_features']
          data_min = self.params.get("data_min", 1e-2)

        if self._mel_basis is None or self._both:
          htk = True
          norm = None
          if self.params.get('mel_type', 'htk') == 'slaney':
            htk = False
            norm = 1

          spectrogram = get_mel(
              mag_spectrogram,
              fs=self._sampling_rate,
              n_fft=self._n_fft,
              power=mag_power,
              feature_normalize=self.params["feature_normalize"],
              mean=self.params.get("feature_normalize_mean", 0.),
              std=self.params.get("feature_normalize_std", 1.),
              mel_basis=self._mel_basis,
              n_mels=n_mels,
              data_min=data_min,
              htk=htk,
              norm=norm
          )
        else:
          spectrogram = mag_spectrogram
      if not self._mel or self._both:
        if self._both:
          num_feats = self.params['num_audio_features']['magnitude']
          data_min = self.params.get("data_min", 1e-5)
          if isinstance(data_min, dict):
            data_min = data_min["magnitude"]
          mel_spectrogram = spectrogram
        else:
          num_feats = self.params['num_audio_features']
          data_min = self.params.get("data_min", 1e-5)

        if mag_power != 1:
          mag_spectrogram = mag_spectrogram * mag_power
          mag_spectrogram = np.clip(
              mag_spectrogram,
              a_min=np.log(data_min),
              a_max=None
          )
        # Else it is a magnitude spec, and we need to normalize
        if self.params["feature_normalize"]:
          mag_spectrogram = self._normalize(mag_spectrogram)
        spectrogram = mag_spectrogram[:, :num_feats]
        if self._both and self._exp_mag:
          spectrogram = np.exp(spectrogram)
      # if self._both:
      #   spectrogram = np.concatenate((mel_spectrogram, spectrogram), axis=1)
    else:
      if self.params.get('dataset_location', None) is not None:
        if "wavs" in audio_filename:
          file_path = os.path.join(
              self.params['dataset_location'], audio_filename + ".wav"
          )
        else:
          file_path = os.path.join(
              self.params['dataset_location'], "wavs", audio_filename + ".wav"
          )
      else:
        file_path = os.path.join(audio_filename)
      if self._mel:
        features_type = "mel_htk"
        if self.params.get('mel_type', 'htk') == 'slaney':
          features_type = "mel_slaney"
      else:
        features_type = self.params['output_type']

      num_feats = self.params['num_audio_features']
      if self._mel_basis is not None and not self._both:
        features_type = "magnitude"
        num_feats = self._n_fft // 2 + 1
      spectrogram = get_speech_features_from_file(
          file_path,
          num_feats,
          features_type=features_type,
          n_fft=self._n_fft,
          mag_power=self.params.get('mag_power', 2),
          feature_normalize=self.params["feature_normalize"],
          mean=self.params.get("feature_normalize_mean", 0.),
          std=self.params.get("feature_normalize_std", 1.),
          trim=self.params.get("trim", False),
          data_min=self.params.get("data_min", 1e-5)
      )
      if self._both:
        mel_spectrogram, spectrogram = spectrogram
        if self._exp_mag:
          spectrogram = np.exp(spectrogram)
    stop_token_target = np.zeros(
        [len(spectrogram)], dtype=self.params['dtype'].as_numpy_dtype()
    )
    if self.params.get("pad_EOS", True):
      # num_pad = pad_to - (len(spectrogram) % pad_to)
      num_pad = pad_to - ((len(spectrogram) + 1) % pad_to) + 1

      data_min = self.params.get("data_min", 1e-5)
      if isinstance(data_min, dict):
        pad_value_mel = self.params.get("pad_value", np.log(data_min["mel"]))
        if self._exp_mag:
          pad_value_mag = self.params.get("pad_value", data_min["magnitude"])
        else:
          pad_value_mag = self.params.get("pad_value", np.log(data_min["magnitude"]))
        if self.params["feature_normalize"]:
          pad_value_mel = self._normalize(pad_value_mel)
          pad_value_mag = self._normalize(pad_value_mag)
      else:
        pad_value = self.params.get("pad_value", np.log(data_min))
        if self.params["feature_normalize"]:
          pad_value = self._normalize(pad_value)
        pad_value_mel = pad_value_mag = pad_value

      if self._both:
        mel_spectrogram = np.pad(
            mel_spectrogram,
            # ((8, num_pad), (0, 0)),
            ((0, num_pad), (0, 0)),
            "constant",
            constant_values=pad_value_mel
        )
        spectrogram = np.pad(
            spectrogram,
            # ((8, num_pad), (0, 0)),
            ((0, num_pad), (0, 0)),
            "constant",
            constant_values=pad_value_mag
        )
        spectrogram = np.concatenate((mel_spectrogram, spectrogram), axis=1)
      else:
        spectrogram = np.pad(
            spectrogram,
            # ((8, num_pad), (0, 0)),
            ((0, num_pad), (0, 0)),
            "constant",
            constant_values=pad_value
        )
      # stop_token_target = np.pad(
      #     stop_token_target, ((8, 0)), "constant", constant_values=0
      # )
      stop_token_target = np.pad(
          stop_token_target, ((0, num_pad)), "constant", constant_values=1
      )
    else:
      stop_token_target[-1] = 1.

    if self._mel_basis is not None:
      mel_basis = self._mel_basis.T.astype(self.params['dtype'].as_numpy_dtype())
    else:
      mel_basis = np.array([0.]).astype(self.params['dtype'].as_numpy_dtype())

    assert len(text_input) % pad_to == 0
    assert len(spectrogram) % pad_to == 0
    return np.int32(text_input), \
           np.int32([len(text_input)]), \
           spectrogram.astype(self.params['dtype'].as_numpy_dtype()), \
           stop_token_target.astype(self.params['dtype'].as_numpy_dtype()), \
           np.int32([len(spectrogram)]), \
           mel_basis

  def _parse_transcript_element(self, transcript):
    """Parses text from file and returns array of text features.

    Args:
      transcript: the string to parse.

    Returns:
      tuple: target text as `np.array` of ids, target text length.
    """
    if six.PY2:
      transcript = unicode(transcript, "utf-8")
    elif not isinstance(transcript, string_types):
      transcript = str(transcript, "utf-8")
      
    if self.params.get("text_cleaners", None):
      # Run Keith Ito's text preprocessing
      text_input = text_to_sequence(transcript, ["english_cleaners"])
    else:
      transcript = transcript.lower()
      text_input = np.array(
          [self.params['char2idx'].get(c,3) for c in transcript]
      )
      pad_to = self.params.get('pad_to', 8)
      if self.params.get("pad_EOS", True):
        # num_pad = pad_to - (len(text_input) % pad_to)
        num_pad = pad_to - ((len(text_input) + 2) % pad_to)
        text_input = np.pad(
            text_input, ((1, 1)),
            "constant",
            constant_values=(
                (self.params['char2idx']["<s>"], self.params['char2idx']["</s>"])
            )
        )
        text_input = np.pad(
            text_input, ((0, num_pad)),
            "constant",
            constant_values=self.params['char2idx']["<p>"]
        )

    return np.int32(text_input), \
           np.int32([len(text_input)])

  def parse_text_output(self, text):
    if self.params.get("text_cleaners", None):
      # Run Keith Ito's text preprocessing
      text = sequence_to_text(text)
      # text = text.replace("_","")
    else:
      text = "".join([self.params['idx2char'][k] for k in text])
      # text = text.replace("<p>","")
      # text = text.replace("</s>","~")
      # text = text.replace("<s>","")
    return text

  def create_interactive_placeholders(self):
    self._text = tf.placeholder(
        dtype=tf.int32,
        shape=[self.params["batch_size"], None]
    )
    self._text_length = tf.placeholder(
        dtype=tf.int32,
        shape=[self.params["batch_size"]]
    )

    self._input_tensors = {}
    self._input_tensors["source_tensors"] = [self._text, self._text_length]

    if self.params.get("style_input", None) == "token":
      self._tokens = tf.placeholder(
          dtype=tf.float32,
          shape=[self.params["batch_size"], 32]
      )
      self._input_tensors["source_tensors"].extend([self._tokens])

  def create_feed_dict(self, model_in):
    """ Creates the feed dict for interactive infer

    Args:
      model_in (str): The string to be spoken.

    Returns:
      feed_dict (dict): Dictionary with values for the placeholders.
    """
    if isinstance(model_in, list):
      tokens = model_in[1]
      model_in = model_in[0]
    if not isinstance(model_in, string_types):
      raise ValueError(
          "Text2Speech's interactive inference mode only supports string.",
          "Got {}". format(type(model_in))
      )
    text, text_length = self._parse_transcript_element(model_in)

    text = np.reshape(text, [self.params["batch_size"], -1])
    text_length = np.reshape(text_length, [self.params["batch_size"]])

    feed_dict = {
        self._text: text,
        self._text_length: text_length,
    }

    if self.params.get("style_input", None) == "token":
      tokens = np.reshape(tokens , [self.params["batch_size"], 32])
      feed_dict[self._tokens] = tokens
    return feed_dict

  @property
  def input_tensors(self):
    return self._input_tensors

  @property
  def sampling_rate(self):
    return self._sampling_rate

  @property
  def n_fft(self):
    return self._n_fft

  def get_size_in_samples(self):
    """Returns the number of audio files."""
    return len(self._files)

  def get_magnitude_spec(self, spectrogram, is_mel=None):
    """Returns an energy magnitude spectrogram. The processing depends on the
    data leyer params.

    Args:
      spectrogram: output spec from model

    Returns:
      mag_spec: mag spec
    """
    spectrogram = spectrogram.astype(float)
    ### No longer needed
    # If outputting both, just trim the mel part and revert the magnitude portion
    # if self._both:
    #   spectrogram = spectrogram[:, self.params['num_audio_features']['mel']:]
    if self._mel or (is_mel and self._both):
      htk = True
      norm = None
      if self.params.get('mel_type', 'htk') == 'slaney':
        htk = False
        norm = 1
      n_feats = self.params['num_audio_features']
      if self._both:
        n_feats = n_feats["mel"]
      return inverse_mel(
          spectrogram,
          fs=self._sampling_rate,
          n_fft=self._n_fft,
          n_mels=n_feats,
          power=self.params.get('mag_power', 2),
          feature_normalize=self.params["feature_normalize"],
          mean=self.params.get("feature_normalize_mean", 0.),
          std=self.params.get("feature_normalize_std", 1.),
          mel_basis=self._mel_basis,
          htk=htk,
          norm=norm
      )
    # Else it is a mag spec
    else:
      if self.params["feature_normalize"]:
        spectrogram = self._denormalize(spectrogram)
      n_feats = self.params['num_audio_features']
      data_min = self.params.get("data_min", 1e-5)
      if self._both:
        n_feats = n_feats["magnitude"]
        if isinstance(data_min, dict):
          data_min = data_min["magnitude"]
        if not self._exp_mag:
          data_min = np.log(data_min)
      else:
        data_min = np.log(data_min)
      # Ensure that num_features is consistent with n_fft
      if n_feats < self._n_fft // 2 + 1:
        num_pad = (self._n_fft // 2 + 1) - spectrogram.shape[1]
        spectrogram = np.pad(
            spectrogram,
            ((0, 0), (0, num_pad)),
            "constant",
            constant_values=data_min
        )
      mag_spec = spectrogram * 1.0 / self.params.get('mag_power', 2)
      if not self._both and not self._exp_mag:
        mag_spec = np.exp(mag_spec)
      return mag_spec

  def _normalize(self, spectrogram):
    return normalize(
        spectrogram,
        mean=self.params.get("feature_normalize_mean", 0.),
        std=self.params.get("feature_normalize_std", 1.)
    )

  def _denormalize(self, spectrogram):
    return denormalize(
        spectrogram,
        mean=self.params.get("feature_normalize_mean", 0.),
        std=self.params.get("feature_normalize_std", 1.)
    )
