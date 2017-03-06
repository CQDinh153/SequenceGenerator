import tensorflow.contrib.rnn as rnn
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from CustomLSTM import MyLSTM
import tensorflow as tf
import numpy as np
import time
import os


class DataLoader(ABC):
    def __init__(self, filepath, batches_folder, vocab):
        self.data = filepath
        self.dir = batches_folder
        self.vocab = vocab

    # Prints out the data that a sequence of integers represents
    @abstractmethod
    def print(self, seq):
        pass

    # Returns a file object pointing to the file at the path for reading
    @abstractmethod
    def open_file(self, path):
        pass

    # Reads a single value from the file object
    @abstractmethod
    def read_value(self, file):
        pass

    # Resets the pointer of a file object to the beginning
    @abstractmethod
    def reset_file(self, file):
        pass

    # Closes the file object
    @abstractmethod
    def close_file(self, file):
        pass

    # Counts the total usable values that can be read from the file object
    # Resets the file pointer of the object as a side effect
    def count_values(self, file):
        self.reset_file(file)
        data = self.read_value(file)
        count = 0
        while data and data in self.vocab:
            count += 1
            data = self.read_value(file)
        self.reset_file(file)
        return count

    def preprocess(self, batch_size):
        data_file = self.open_file(self.data)
        num_values = self.count_values(data_file)
        row_length = (num_values - 1) // batch_size

        filename = os.path.basename(self.data)
        file_dir = os.path.join(self.dir, filename)
        batches_dir = os.path.join(file_dir, str(batch_size))
        in_dir = os.path.join(batches_dir, "in")
        out_dir = os.path.join(batches_dir, "out")
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        if not os.path.exists(batches_dir):
            os.mkdir(batches_dir)
        if not os.path.exists(in_dir):
            os.mkdir(in_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        self.reset_file(data_file)
        for batch_num in range(batch_size):
            in_path = os.path.join(in_dir, str(batch_num))
            with open(in_path, "w", encoding="ascii") as input_file:
                for i in range(row_length):
                    data = self.vocab[self.read_value(data_file)]
                    input_file.write(str(data) + "\n")
        self.reset_file(data_file)
        self.read_value(data_file)
        for batch_num in range(batch_size):
            out_path = os.path.join(out_dir, str(batch_num))
            with open(out_path, "w", encoding="ascii") as input_file:
                for i in range(row_length):
                    data = self.vocab[self.read_value(data_file)]
                    input_file.write(str(data) + "\n")
        self.close_file(data_file)

    def batch_files(self, batch_size):
        filename = os.path.basename(self.data)
        file_dir = os.path.join(self.dir, filename)
        batches_dir = os.path.join(file_dir, str(batch_size))
        in_dir = os.path.join(batches_dir, "in")
        out_dir = os.path.join(batches_dir, "out")
        if not (os.path.exists(in_dir) and os.path.exists(out_dir)):
            self.preprocess(batch_size)

        batch_files = []
        for batch_num in range(batch_size):
            in_path = os.path.join(in_dir, str(batch_num))
            out_path = os.path.join(out_dir, str(batch_num))
            in_file = open(in_path, encoding="ascii")
            out_file = open(out_path, encoding="ascii")
            batch_files.append((in_file, out_file))
        return batch_files

    def num_batches(self, batch_size, sequence_length):
        count = 0
        for batch in self.batches(batch_size, sequence_length):
            count += 1
        return count

    def batches(self, batch_size, sequence_length):
        files = self.batch_files(batch_size)

        done = False
        while not done:
            input_data = []
            target_data = []
            for row in range(batch_size):
                in_file, out_file = files[row]
                input_sequence = []
                target_sequence = []
                for i in range(sequence_length):
                    in_data = in_file.readline()
                    out_data = out_file.readline()
                    if in_data and out_data:
                        input_sequence.append(int(in_data))
                        target_sequence.append(int(out_data))
                if len(input_sequence) == len(target_sequence) == sequence_length:
                    input_data.append(input_sequence)
                    target_data.append(target_sequence)
            input_data = np.array(input_data)
            target_data = np.array(target_data)
            if input_data.shape == target_data.shape == (batch_size, sequence_length):
                yield input_data, target_data
            else:
                done = True

        for in_file, out_file in files:
            self.reset_file(in_file)
            self.reset_file(out_file)

        return np.array([]), np.array([])


class CharLoader(DataLoader):
    def __init__(self, filepath, batches_folder, vocab=None, encoding="utf8"):
        DataLoader.__init__(self, filepath, batches_folder, vocab)
        self.encoding = encoding
        if vocab is None:
            self.vocab = {}
            self.load_vocab(filepath)

    def print(self, seq):
        reverse_vocab = {self.vocab[key]: key for key in self.vocab}
        for value in seq:
            print(reverse_vocab[value], end="")
        print()

    def load_vocab(self, path):
        chars = set(list(self.vocab))
        with open(path, encoding=self.encoding) as file:
            value = file.read(1)
            while value:
                chars.add(value)
                value = file.read(1)
        chars = sorted(list(chars))
        self.vocab = {}
        for i in range(len(chars)):
            self.vocab[chars[i]] = i

    def save_vocab(self, path):
        chars = sorted(list(self.vocab))
        with open(path, "w", encoding=self.encoding) as file:
            for char in chars:
                file.write(char)

    # Returns a file object pointing to the file at the path for reading
    def open_file(self, path):
        return open(path, encoding=self.encoding)

    # Reads a single value from the file object
    def read_value(self, file):
        return file.read(1)

    # Resets the pointer of a file object to the beginning
    def reset_file(self, file):
        file.seek(0)

    # Closes the file object
    def close_file(self, file):
        file.close()


def in_alphabet(char):
    if ord(char) > 122:
        return False
    if ord(char) < 65:
        return False
    if 90 < ord(char) < 97:
        return False
    return True


class WordLoader(DataLoader):
    def __init__(self, filepath, batches_folder, vocab=None, encoding="utf8"):
        DataLoader.__init__(self, filepath, batches_folder, vocab)
        self.encoding = encoding
        if vocab is None:
            self.vocab = {}
            self.load_vocab(filepath)

    def print(self, seq):
        reverse_vocab = {self.vocab[key]: key for key in self.vocab}
        for value in seq:
            if value in reverse_vocab:
                word = reverse_vocab[value]
                print(word, end='')

        print()

    def load_vocab(self, path):
        words = set(list(self.vocab))
        with open(path, encoding=self.encoding) as file:
            value = self.read_value(file)
            while value:
                words.add(value)
                value = self.read_value(file)
        words = sorted(list(words))
        self.vocab = {}
        for i in range(len(words)):
            self.vocab[words[i]] = i

    def save_vocab(self, path):
        words = sorted(list(self.vocab))
        with open(path, "w", encoding=self.encoding) as file:
            for word in words:
                file.write(word)

    # Returns a file object pointing to the file at the path for reading
    def open_file(self, path):
        return open(path, encoding=self.encoding)

    # Reads a single value from the file object
    def read_value(self, file):
        value = ""
        char = file.read(1)
        while char and char != " " and char != "\n" and char != "\t":
            value += char
            char = file.read(1)
        value += char
        return value

    # Resets the pointer of a file object to the beginning
    def reset_file(self, file):
        file.seek(0)

    # Closes the file object
    def close_file(self, file):
        file.close()


class SequenceLearner:
    def __init__(self, vocab_size, rnn_size, num_layers, embedding_size, int_type=tf.int32, float_type=tf.float32):
        self.softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
        self.softmax_b = tf.get_variable("softmax_b", [vocab_size])
        self.embedding = tf.get_variable("embedding", [vocab_size, embedding_size])

        self.embedding_size = embedding_size
        self.depth = num_layers
        self.width = rnn_size
        self.vocab_size = vocab_size

        self.train_cells = []
        self.test_cells = []
        for i in range(num_layers):
            train = MyLSTM(rnn_size, state_is_tuple=True)
            test = MyLSTM(rnn_size, state_is_tuple=True)
            self.train_cells.append(train)
            self.test_cells.append(test)
        self.train_cell = rnn.MultiRNNCell(self.train_cells, state_is_tuple=True)
        self.test_cell = rnn.MultiRNNCell(self.test_cells, state_is_tuple=True)

        self.int_type = int_type
        self.float_type = float_type

        self.batch_shape = (-1, -1)
        self.train_input = None
        self.train_state_i = None
        self.train_state_f = None
        self.train_results = None
        self.train_targets = None
        self.train_cost = None
        self.learning_rate = None
        self.train_op = None

        self.test_input = None
        self.test_state_i = None
        self.test_results = None
        self.test_state_f = None

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.test_state = None
        self.train_state = None

    def reset_train(self, batch_size):
        self.train_state = self.sess.run(self.train_cell.zero_state(batch_size, self.float_type))

    def reset_test(self):
        self.test_state = self.sess.run(self.test_cell.zero_state(1, self.float_type))

    def train_setup(self, batch_size, sequence_length, grad_clip, parallel_iterations=32, initial_learning_rate=0):
        # A placeholder for the input tensor
        input_data = tf.placeholder(self.int_type, [batch_size, sequence_length])

        embedded = tf.nn.embedding_lookup(self.embedding, input_data)

        # The initial state should just be zeros
        initial_state = self.train_cell.zero_state(batch_size, self.float_type)

        # The results of the unrolled network
        outputs, final_state = tf.nn.dynamic_rnn(self.train_cell, embedded, None, initial_state, parallel_iterations=parallel_iterations)

        # Shape the outputs so they can be run through the softmax layer
        output = tf.reshape(outputs, [-1, self.width])

        # Run the output through the softmax layer
        logits = tf.matmul(output, self.softmax_w) + self.softmax_b

        logits = tf.reshape(logits, [batch_size, sequence_length, self.vocab_size])

        # The final output of the whole network
        results = tf.nn.softmax(logits)

        # A placeholder for the target tensor
        targets = tf.placeholder(self.int_type, [batch_size, sequence_length])

        # The loss over the sequence
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

        # The cost function
        cost = tf.reduce_sum(loss) / batch_size / sequence_length

        # The learning rate
        learning_rate = tf.Variable(initial_learning_rate, trainable=False)

        # Get the training operation
        vars = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, vars),
                                          grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        op = optimizer.apply_gradients(zip(grads, vars))

        self.batch_shape = (batch_size, sequence_length)
        self.train_input = input_data
        self.train_state_i = initial_state
        self.train_state_f = final_state
        self.train_results = results
        self.train_targets = targets
        self.train_cost = cost
        self.learning_rate = learning_rate
        self.train_op = op

        for i in range(self.depth):
            self.test_cells[i].link_weights(self.train_cells[i])

        self.reset_train(batch_size)
        self.test_setup()

    def test_setup(self):
        # A placeholder for the input tensor
        input_data = tf.placeholder(self.int_type, [1, 1])

        embedded = tf.nn.embedding_lookup(self.embedding, input_data)

        # The initial state should just be zeros
        initial_state = self.test_cell.zero_state(1, self.float_type)

        # The results of the unrolled network
        outputs, final_state = tf.nn.dynamic_rnn(self.test_cell, embedded, None, initial_state, 1)

        # The final output of the LSTM
        output = tf.reshape(outputs, outputs.shape[1:])

        # Run the output through the softmax layer
        logits = tf.matmul(output, self.softmax_w) + self.softmax_b

        # The final output of the whole network
        results = tf.nn.softmax(logits)

        self.test_input = input_data
        self.test_state_i = initial_state
        self.test_results = results
        self.test_state_f = final_state

        self.reset_test()

        self.sess.run(tf.global_variables_initializer())

    def save_model(self, model_dir):
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        for layer in range(self.depth):
            cell = self.train_cells[layer]
            W = self.sess.run(cell.W)
            b = self.sess.run(cell.b)
            weight_filename = os.path.join(model_dir, "W_{}.txt".format(layer))
            bias_filename = os.path.join(model_dir, "b_{}.txt".format(layer))
            np.savetxt(weight_filename, W, "%10.10f")
            np.savetxt(bias_filename, b, "%10.10f")

        softmax_weight_filename = os.path.join(model_dir, "softmax_W.txt")
        softmax_bias_filename = os.path.join(model_dir, "softmax_b.txt")
        embedding_filename = os.path.join(model_dir, "embedding.txt")

        np.savetxt(softmax_weight_filename, self.sess.run(self.softmax_w), "%10.10f")
        np.savetxt(softmax_bias_filename, self.sess.run(self.softmax_b), "%10.10f")
        np.savetxt(embedding_filename, self.sess.run(self.embedding), "%10.10f")

    # Inputs:
    # model_dir: The directory that the trained values are stored in
    def load_model(self, model_dir):
        self.sess.run(tf.global_variables_initializer())

        softmax_weight_filename = os.path.join(model_dir, "softmax_W.txt")
        softmax_bias_filename = os.path.join(model_dir, "softmax_b.txt")
        embedding_filename = os.path.join(model_dir, "embedding.txt")

        if not os.path.exists(softmax_weight_filename):
            return False
        if not os.path.exists(embedding_filename):
            return False
        if not os.path.exists(softmax_bias_filename):
            return False

        self.sess.run(tf.assign(self.softmax_w, np.loadtxt(softmax_weight_filename)))
        self.sess.run(tf.assign(self.softmax_b, np.loadtxt(softmax_bias_filename)))
        self.sess.run(tf.assign(self.embedding, np.loadtxt(embedding_filename)))

        for layer in range(self.depth):
            weight_filename = os.path.join(model_dir, "W_{}.txt".format(layer))
            bias_filename = os.path.join(model_dir, "b_{}.txt".format(layer))
            if not os.path.exists(weight_filename):
                return False
            if not os.path.exists(bias_filename):
                return False
            w = np.loadtxt(weight_filename)
            b = np.loadtxt(bias_filename)
            cell = self.train_cells[layer]
            cell.set_weights(self.sess, w, b)

        for i in range(self.depth):
            self.test_cells[i].link_weights(self.train_cells[i])

        return True

    def sample_probs(self, probabilities):
        probabilities = probabilities[0]
        cumulative_probabilities = np.cumsum(probabilities)
        sample = np.random.rand(1) * np.sum(probabilities)
        return int(np.searchsorted(cumulative_probabilities, sample))

    def test(self, val):
        x = np.zeros((1, 1))
        x[0, 0] = val
        feed = {self.test_input: x, self.test_state_i: self.test_state}
        results, self.test_state = self.sess.run([self.test_results, self.test_state_f], feed)
        return self.sample_probs(results)

    def train_batch(self, input_batch, target_batch):
        input_batch = np.array(input_batch)
        target_batch = np.array(target_batch)

        assert self.train_op is not None, "Training parameters must be set up"
        assert input_batch.shape == self.batch_shape, "Incorrect batch shape. Expected {}, got {}".format(self.batch_shape, input_batch.shape)
        assert target_batch.shape == self.batch_shape, "Incorrect batch shape. Expected {}, got {}".format(self.batch_shape, target_batch.shape)

        feed = {self.train_input: input_batch, self.train_targets: target_batch}
        for i, (c, h) in enumerate(self.train_state_i):
            feed[c] = self.train_state[i].c
            feed[h] = self.train_state[i].h

        loss, self.train_state, status = self.sess.run([self.train_cost, self.train_state_f, self.train_op], feed)

        return loss

    def sample(self, sample_size, sample_seed=None):
        if sample_seed is None:
            sample_seed = [0]

        self.reset_test()

        result = []
        for val in sample_seed[:-1]:
            self.test(val)

        val = sample_seed[-1]
        for n in range(sample_size):
            val = self.test(val)
            result.append(val)
        return result

    def train(self, loader, model_dir, batch_size, sequence_length, num_epochs, grad_clip, learning_rate, decay_rate=1, decay_after=0, sample_size=0, sample_every=0, sample_seed=None, parallel_iterations=32, plot=False):
        self.train_setup(batch_size, sequence_length, grad_clip, parallel_iterations, learning_rate)

        self.load_model(model_dir)
        self.save_model(model_dir)

        num_batches = loader.num_batches(batch_size, sequence_length)
        total_batches = num_epochs * num_batches
        times = []
        losses = [float('inf')]
        if plot:
            plt.ion()
        for epoch in range(num_epochs):
            batches = 0
            epoch_start = time.time()
            losses = [losses[-1]]
            if epoch >= decay_after:
                self.sess.run(tf.assign(self.learning_rate, learning_rate * (decay_rate ** (epoch - decay_after))))

            if sample_every > 0 and not epoch % sample_every:
                loader.print(self.sample(sample_size, sample_seed))

            for input_batch, target_batch in loader.batches(batch_size, sequence_length):
                start = time.time()
                loss = self.train_batch(input_batch, target_batch)
                end = time.time()

                losses.append(loss)
                batches += 1
                times.append(end - start)
                times = times[-batch_size:]
                print("Epoch {}: {}/{} - {:.2f}%, train_loss = {:.3f}, learning_rate = {}, time/batch = {:.3f}, ETA = {:.3f}".format(
                    epoch,
                    batches,
                    num_batches,
                    100 * batches / num_batches,
                    loss,
                    self.sess.run(self.learning_rate),
                    times[-1],
                    (sum(times) / len(times)) * (total_batches - batches - num_batches * epoch))
                )
            epoch_end = time.time()
            self.save_model(model_dir)
            print("Epoch {} of {}:".format(epoch, num_epochs))
            print("Inital Loss: {}".format(losses[0]))
            print("Final Loss: {}".format(losses[-1]))
            print("Took: {} seconds".format(epoch_end - epoch_start))
            if plot and len(losses) > 1:
                plt.scatter([epoch], [sum(losses[1:]) / len(losses[1:])], c="b")
                plt.pause(0.05)
        if plot:
            plt.ioff()
            plt.show()


class CharModel:
    def __init__(self, rnn_size, num_layers, embedding_size, model_dir, vocab_file, encoding='utf8', int_type=tf.int32, float_type=tf.float32):
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.dir = model_dir
        self.encoding = encoding
        assert os.path.exists(vocab_file), "Vocab file does not exist"
        self.vocab = {}
        self.load_vocab(vocab_file)
        self.learner = SequenceLearner(len(self.vocab), rnn_size, num_layers, embedding_size, int_type, float_type)

    def load_vocab(self, path):
        chars = set(list(self.vocab))
        with open(path, encoding=self.encoding) as file:
            value = file.read(1)
            while value:
                chars.add(value)
                value = file.read(1)
        chars = sorted(list(chars))
        self.vocab = {}
        for i in range(len(chars)):
            self.vocab[chars[i]] = i

    def save_vocab(self, path):
        chars = sorted(list(self.vocab))
        with open(path, "w", encoding=self.encoding) as file:
            for char in chars:
                file.write(char)

    def train(self, input_data, batches_folder, batch_size, sequence_length, num_epochs, learning_rate, decay_rate=1, decay_after=0, grad_clip=5, sample_size=0, sample_every=0, sample_seed=None, parallel_iterations=32, plot=False):
        if isinstance(input_data, list):
            for filepath in input_data:
                self.train(filepath, batches_folder, batch_size, sequence_length, num_epochs, grad_clip, learning_rate, decay_rate, decay_after, sample_size, sample_every, sample_seed, parallel_iterations, plot)
        loader = CharLoader(input_data, batches_folder, self.vocab, self.encoding)
        sample_seed = [self.vocab[ch] for ch in list(sample_seed)]
        self.learner.train(loader, self.dir, batch_size, sequence_length, num_epochs, grad_clip, learning_rate, decay_rate, decay_after, sample_size, sample_every, sample_seed, parallel_iterations, plot)

    def sample(self, sample_size, sample_seed):
        sample_seed = [self.vocab[ch] for ch in list(sample_seed)]
        self.learner.sample(sample_size, sample_seed)


class WordModel:
    def __init__(self, rnn_size, num_layers, embedding_size, model_dir, vocab_file, encoding='utf8', int_type=tf.int32, float_type=tf.float32):
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.dir = model_dir
        self.encoding = encoding
        assert os.path.exists(vocab_file), "Vocab file does not exist"
        self.vocab = {' ', '\n', '\t'}
        self.load_vocab(vocab_file)
        self.learner = SequenceLearner(len(self.vocab), rnn_size, num_layers, embedding_size, int_type, float_type)

    def load_vocab(self, path):
        chars = set(list(self.vocab))
        with open(path, encoding=self.encoding) as file:
            value = self.read_word(file)
            while value:
                chars.add(value)
                value = self.read_word(file)
        chars = sorted(list(chars))
        self.vocab = {}
        for i in range(len(chars)):
            self.vocab[chars[i]] = i

    def read_word(self, file):
        value = ""
        char = file.read(1)
        while char != " " and char != "\n" and char != "\t" and char:
            value += char
            char = file.read(1)
        value += char
        return value

    def save_vocab(self, path):
        chars = sorted(list(self.vocab))
        with open(path, "w", encoding=self.encoding) as file:
            for char in chars:
                file.write(char)

    def train(self, input_data, batches_folder, batch_size, sequence_length, num_epochs, learning_rate, decay_rate=1, decay_after=0, grad_clip=5, sample_size=0, sample_every=0, sample_seed=None, parallel_iterations=32, plot=False):
        if isinstance(input_data, list):
            for filepath in input_data:
                self.train(filepath, batches_folder, batch_size, sequence_length, num_epochs, grad_clip, learning_rate, decay_rate, decay_after, sample_size, sample_every, sample_seed, parallel_iterations, plot)
        loader = WordLoader(input_data, batches_folder, self.vocab, self.encoding)
        sample_seed = [self.vocab[ch] for ch in list(sample_seed)]
        self.learner.train(loader, self.dir, batch_size, sequence_length, num_epochs, grad_clip, learning_rate, decay_rate, decay_after, sample_size, sample_every, sample_seed, parallel_iterations, plot)

    def sample(self, sample_size, sample_seed):
        sample_seed = [self.vocab[ch] for ch in list(sample_seed)]
        self.learner.sample(sample_size, sample_seed)
