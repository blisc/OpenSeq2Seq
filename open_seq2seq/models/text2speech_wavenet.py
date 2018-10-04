# Copyright (c) 2018 NVIDIA Corporation
import tensorflow as tf
from scipy.io.wavfile import write
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from .encoder_decoder import EncoderDecoderModel

def plot_spectrograms(
    inputs,
    outputs,
    logdir,
    train_step,
    number=0,
    append=False,
):
  fig = plt.figure(figsize=(8, 4))

  inputs = inputs.astype(float)
  plt.plot(inputs, 'r', linewidth=.1)
  outputs = outputs.astype(float)
  plt.plot(outputs, 'g', linewidth=.1)

  plt.xlabel('time')
  plt.tight_layout()

  if append:
    name = '{}/Output_step{}_{}_{}.png'.format(
        logdir, train_step, number, append
    )
  else:
    name = '{}/Output_step{}_{}.png'.format(logdir, train_step, number)
  if logdir[0] != '/':
    name = "./" + name
  #save
  fig.savefig(name, dpi=300)

  plt.close(fig)


def save_audio(signal, logdir, step, sampling_rate, mode):
  file_name = '{}/sample_step{}_{}.wav'.format(logdir, step, mode)
  if logdir[0] != '/':
    file_name = "./" + file_name
  write(file_name, sampling_rate, signal)

class Text2SpeechWavenet(EncoderDecoderModel):
  # [TODO] add logging info

  @staticmethod
  def get_required_params():
    return dict(
      EncoderDecoderModel.get_required_params(), **{
        # "key": int,
      }
    )

  def __init__(self, params, mode="train", hvd=None):
    super(Text2SpeechWavenet, self).__init__(params, mode=mode, hvd=hvd)

  def maybe_print_logs(self, input_values, output_values, training_step):
    encoded_inputs, audio, encoded_outputs = output_values
    plot_spectrograms(
        encoded_inputs[-1][::100], encoded_outputs[-1][::100], self.params["logdir"], training_step)
    save_audio(
        audio[-1],
        self.params["logdir"],
        training_step,
        sampling_rate=22050,
        mode="train"
    )
    return {}

  def evaluate(self, input_values, output_values):
    return output_values[1][-1]

  def finalize_evaluation(self, results_per_batch, training_step=None):
    save_audio(
        results_per_batch[0],
        self.params["logdir"],
        training_step,
        sampling_rate=22050,
        mode="eval"
    )
    return {}
    
