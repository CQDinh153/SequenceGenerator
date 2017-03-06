from Model import CharModel, WordModel
import tensorflow as tf
import os

rnn_size = 128
num_layers = 2
embedding_size = rnn_size
model_dir = "model"
vocab_file = os.path.join(model_dir, "vocab")
encoding = None
int_type = tf.int32
float_type = tf.float32

input_file = "data/shakespeare.txt"
batches_folder = "data_processed"
batch_size = 50
sequence_length = 100
num_epochs = 200
learning_rate = 0.002
decay_rate = 0.97
decay_after = 10
grad_clip = 5
sample_size = 200
sample_every = 1
sample_seed = " "
parallel_iterations = batch_size
plot = True

get_vocab = False
input_type = 'word'

model_class_dict = {'char':CharModel, 'word':WordModel}

with tf.device("/gpu:0"):
    model_class = model_class_dict[input_type]
    if get_vocab:
        trainer = model_class(rnn_size, num_layers, embedding_size, model_dir, input_file, encoding, int_type, float_type)
        trainer.save_vocab(vocab_file)
    else:
        trainer = model_class(rnn_size, num_layers, embedding_size, model_dir, vocab_file, encoding, int_type, float_type)
    trainer.train(input_file, batches_folder, batch_size, sequence_length, num_epochs, learning_rate, decay_rate, decay_after, grad_clip, sample_size, sample_every, sample_seed, parallel_iterations, plot)