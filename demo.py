import librosa
# import IPython
import time

import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf

from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
                                     create_logdir, create_model, get_interactive_infer_results
from open_seq2seq.models.text2speech import save_audio

batch_size = 8
input_file = "Infer_T2T/train.clean.de"
print_every = 10
print_verbose = False

args_S2T = ["--config_file=Infer_S2T/config.py",
        "--mode=interactive_infer",
        "--logdir=Infer_S2T/",
        "--batch_size_per_gpu={}".format(batch_size),
]
args_T2S_en = ["--config_file=Infer_T2S/config.py",
        "--mode=interactive_infer",
        "--logdir=Infer_T2S",
        "--batch_size_per_gpu={}".format(batch_size),
]
args_DE2EN = ["--config_file=Infer_T2T/config.py",
        "--mode=interactive_infer",
        "--logdir=Infer_T2T/",
        "--batch_size_per_gpu={}".format(batch_size),
]
def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(args, base_config, config_module, base_model, None)
    return model, checkpoint

model_S2T, checkpoint_S2T = get_model(args_S2T, "S2T")
model_T2S_en, checkpoint_T2S_en = get_model(args_T2S_en, "T2S_en")
model_DE2EN, checkpoint_DE2EN = get_model(args_DE2EN, "DE2EN")

sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)


def restore_certain_variables(sess, filename, trainables):
    print('Restoring only the variables found in the checkpoint')
#     trainables = {v.name: v for v in tf.trainable_variables()}
    assign_ops = []
    vars_to_initialize = []
    
    try:
        reader = tf.train.NewCheckpointReader(filename)
        var_to_shape_map = reader.get_variable_to_shape_map()
        non_loss_var = {var: var_to_shape_map[var] for var in var_to_shape_map if 'Loss_Optimization' not in var}
        for var in var_to_shape_map:
            if 'global_step' in var:
                print('Restoring from the step', reader.get_tensor(var))
        for name in trainables:
#             print(name)
            idx = name.find(":")
#             print(idx)
            if idx != -1:
                true_name = name[:idx]
            else:
                true_name = name
            # if name.endswith(':0'):
            #     true_name = name[:-2]
#             if true_name in var_to_shape_map and trainables[name].shape == var_to_shape_map[true_name]:
            print('Restoring value to', true_name)
            assign_ops.append(trainables[name].assign(tf.cast(reader.get_tensor(true_name),tf.float32)))
#             if 'EmbeddingMatrix' in true_name:
#                 embed_op, has_embed_op = _restore_embed(trainables[name], var_to_shape_map, reader)
#                 if has_embed_op:
#                     assign_ops.append(embed_op)

        print('assign_ops', assign_ops)
    except Exception as e:    # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                        "with SNAPPY.")
        if ("Data loss" in str(e) and
                (any([e in file_name for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = """
            It's likely that this is a V2 checkpoint and you need to provide the filename
            *prefix*.    Try removing the '.' and extension.    Try:
            inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))
    sess.run(assign_ops)

vars_S2T = {}
vars_T2S_en = {}
vars_DE2EN = {}
for v in tf.get_collection(tf.GraphKeys.VARIABLES):
    if "S2T" in v.name:
        vars_S2T["/".join(v.op.name.split("/")[1:])] = v
    if "T2S_en" in v.name:
        vars_T2S_en["/".join(v.op.name.split("/")[1:])] = v
    if "DE2EN" in v.name:
        vars_DE2EN["/".join(v.op.name.split("/")[1:])] = v
saver_T2S_en = tf.train.Saver(vars_T2S_en)
saver_T2S_en.restore(sess, checkpoint_T2S_en)
# restore_certain_variables(sess, checkpoint_T2S_en, vars_T2S_en)


# saver_S2T = tf.train.Saver(vars_S2T)
# saver_DE2EN = tf.train.Saver(vars_DE2EN)
# saver_S2T.restore(sess, checkpoint_S2T)
# saver_DE2EN.restore(sess, checkpoint_DE2EN)

restore_certain_variables(sess, checkpoint_S2T, vars_S2T)
restore_certain_variables(sess, checkpoint_DE2EN, vars_DE2EN)

import codecs
import os,sys,inspect, re

from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

bpe_vocab_file = "bpe.32000"
with codecs.open(bpe_vocab_file, encoding='utf-8') as bpefile:
    bpe = BPE(bpefile)

def loop_fn(line):
    data_processing_time = 0
    asr_time = 0
    translation_time = 0
    synthesis_time = 0
    start_time = time.time()
    
    if print_verbose:
        print("")
        print("Input German")
        print(line)
    
    model_in = []
    for l in line:
        model_in.append(bpe.process_line(l).encode("utf-8"))
    # Translate to English
#     model_in=bpe.process_line(line)
    results, infer_time = get_interactive_infer_results(model_DE2EN, sess, model_in=model_in)
    end_time = time.time()
    output = results[1]
    new_out = []
    for line in output:
        english_translated = re.sub("&\w+;", "", line)
        english_translated = re.sub("@ ", "", english_translated)
        english_translated = re.sub("/", "", english_translated)
        new_out.append(english_translated)

    if print_verbose:
        print("Translated English")
        print(new_out)
    
    data_processing_time += infer_time-start_time
    translation_time += end_time-infer_time

    # Generate speech
    model_in=new_out
    results, infer_time = get_interactive_infer_results(model_T2S_en, sess, model_in=model_in)
    
    data_processing_time += infer_time-end_time
    
    end_time = time.time()
    prediction = results[1][1]
    audio_length = results[1][4]
    new_out = []
    for pred, audio_len in zip(prediction, audio_length):
        pred = pred[:audio_len-1,:]
        pred = model_T2S_en.get_data_layer().get_magnitude_spec(pred)
        wav = save_audio(pred, "unused", "unused", 22050, save_format="np.array", n_iters=10)
        wav = librosa.core.resample(wav, 22050, 16000)
        new_out.append(wav)
#     audio = IPython.display.Audio(wav, rate=16000)

#     print("Generated Audio")
#     IPython.display.display(audio)
    
    synthesis_time += end_time-infer_time

    # Recognize speech
    model_in = new_out
    results, infer_time = get_interactive_infer_results(model_S2T, sess, model_in=model_in) 
    english_recognized = results[0][0]

    if print_verbose:
        english_recognized += "."
        print("Recognized Speech")
        print(english_recognized)
    
    data_processing_time += infer_time-end_time
    end_time = time.time()
    asr_time = end_time-infer_time
    total_time = end_time-start_time
    
    return total_time, data_processing_time, translation_time, synthesis_time, asr_time

with open(input_file) as f:
    num_read = 0
    arr = []
    i = 0
    total_time = data_processing_time = translation_time = synthesis_time = asr_time = 0
    for line in f:
        arr.append(line)
        num_read += 1
        if num_read == batch_size:
            i += 1
            times = loop_fn(arr)
            total_time_i, data_processing_time_i, translation_time_i, synthesis_time_i, asr_time_i = times
            total_time += total_time_i
            data_processing_time += data_processing_time_i
            translation_time += translation_time_i
            synthesis_time += synthesis_time_i
            asr_time += asr_time_i
            num_read = 0
            arr = []
            if i % print_every == 0:
                print("Done batch {}".format(i*print_every))
                print("total time per batch: {}".format(total_time/i))
                print("data_processing_time per batch: {}".format(data_processing_time/i))
                print("translation_time per batch: {}".format(translation_time/i))
                print("synthesis_time per batch: {}".format(synthesis_time/i))
                print("asr_time per batch: {}".format(asr_time/i))
                print("data_processing_time percentage: {}".format(data_processing_time/total_time))
                print("translation_time percentage: {}".format(translation_time/total_time))
                print("synthesis_time percentage: {}".format(synthesis_time/total_time))
                print("asr_time percentage: {}".format(asr_time/total_time))
    if len(arr) != 0:
        pad = batch_size - len(arr) 
        for i in range(pad):
            arr.append(" ")
        times = loop_fn(arr)

# while True:
#     line = input()
#     if line == "":
#         break
#     IPython.display.clear_output()
#     loop_fn(line)