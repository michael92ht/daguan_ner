# encoding = utf-8

import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers
from .tf_attention import attention_layer


class BatchManager(object):
    def __init__(self, data, is_test=False,  batch_size=128):
        self.is_test = is_test
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.is_test:
            sorted_data = sorted(data, key=lambda x: len(x))
        else:
            sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            tmp = sorted_data[int(i*batch_size): int((i+1)*batch_size)]
            if self.is_test and len(tmp) < batch_size:
                tmp.extend([tmp[0]] * (batch_size - len(tmp)))
            batch_data.append(self.pad_data(tmp))
        return batch_data

    def pad_data(self, data):
        strings = []
        masks = []
        targets = []
        if self.is_test:
            max_length = max([len(sentence) for sentence in data])
        else:
            max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            if self.is_test:
                string = line
                target = [0] * len(line)
            else:
                string, target = line
            string_padding = ['<UNK>'] * (max_length - len(string))
            padding = [0] * (max_length - len(string))
            strings.append(string + string_padding)
            masks.append([1] * len(target) + padding)
            targets.append(target + padding)
        return [strings, masks, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            np.random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


class Model(object):
    def __init__(self, config):
        self.lr = config["lr"]
        self.input_dropout = config["dropout"]
        self.lstm_dim = config["lstm_dim"]
        self.layer_type = config["layer_type"]
        self.use_attention = config["attention"]
        self.num_attention_heads = config['num_attention_heads']
        self.size_per_head = config['size_per_head']
        self.num_tags = 7
        self.char_dim = 300
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # elmo
        self.batcher = TokenBatcher(config['vocab_file'])
        # Input placeholders to the biLM.
        self.context_token_ids = tf.placeholder('int32', shape=(None, None))
        # Build the biLM graph.
        self.bilm = BidirectionalLanguageModel(config['options_file'],
                                               config['weight_file'],
                                               use_character_inputs=False,
                                               embedding_weight_file=config['token_embedding_file']
                                               )
        self.context_embeddings_op = self.bilm(self.context_token_ids)
        self.elmo_context_input = weight_layers('input',
                                                self.context_embeddings_op,
                                                l2_coef=0.0)['weighted_op']

        # add placeholders for the model
        self.mask_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ChatInputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")

        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")
        used = tf.sign(tf.abs(self.mask_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.mask_inputs)[0]
        self.num_steps = tf.shape(self.mask_inputs)[-1]

        self.logits = self.inference(self.elmo_context_input)
        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)
        self.train_op = self.train(self.loss)
        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def loss(self, embedding):
        logits = self.inference(embedding)
        loss = self.loss_layer(logits, self.logits)
        return loss

    def train(self, loss):
        with tf.variable_scope("optimizer"):
            opt = tf.train.AdamOptimizer(self.lr)
            # apply grad clip to avoid gradient explosion
            grads_vars = opt.compute_gradients(loss)
            capped_grads_vars = [[tf.clip_by_value(g, -5, 5), v] for g, v in grads_vars]
            train_op = opt.apply_gradients(capped_grads_vars, self.global_step)
            return train_op

    def single_biLSTM_layer(self, model_inputs, lstm_dim, lengths):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("first_layer"):
            first_fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, state_is_tuple=True)
            # 添加dropout.为了防止过拟合，在它的隐层添加了 dropout 正则
            first_fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(first_fw_lstm_cell, output_keep_prob=self.dropout)
            first_bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, state_is_tuple=True)
            first_bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(first_bw_lstm_cell, output_keep_prob=self.dropout)
            first_outputs, _ = tf.nn.bidirectional_dynamic_rnn(first_fw_lstm_cell,
                                                                        first_bw_lstm_cell,
                                                                        model_inputs,
                                                                        sequence_length=lengths,
                                                                        dtype=tf.float32)
            output = tf.concat(first_outputs, -1)
        return output

    def concat_biLSTM_layer(self, model_inputs, lstm_dim, lengths):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("first_layer"):
            first_fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, state_is_tuple=True)
            # 添加dropout.为了防止过拟合，在它的隐层添加了 dropout 正则
            first_fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(first_fw_lstm_cell, output_keep_prob=self.dropout)
            first_bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, state_is_tuple=True)
            first_bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(first_bw_lstm_cell, output_keep_prob=self.dropout)
            first_outputs, _ = tf.nn.bidirectional_dynamic_rnn(first_fw_lstm_cell,
                                                                        first_bw_lstm_cell,
                                                                        model_inputs,
                                                                        sequence_length=lengths,
                                                                        dtype=tf.float32)
            first_layer_output = tf.concat(first_outputs, -1)

        with tf.variable_scope("second_layer"):
            second_fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, state_is_tuple=True)
            # 添加dropout.为了防止过拟合，在它的隐层添加了 dropout 正则
            second_fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(second_fw_lstm_cell, output_keep_prob=self.dropout)
            second_bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, state_is_tuple=True)
            second_bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(second_bw_lstm_cell, output_keep_prob=self.dropout)
            second_outputs, _ = tf.nn.bidirectional_dynamic_rnn(second_fw_lstm_cell,
                                                                         second_bw_lstm_cell,
                                                                         first_layer_output,
                                                                        sequence_length=lengths,
                                                                        dtype=tf.float32)
            second_layer_output = tf.concat(second_outputs, -1)

        return tf.concat([first_layer_output, second_layer_output], axis=-1)


    def stack_biLSTM_layer(self, model_inputs, lstm_dim, lengths):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        fw_lstms, bw_lstms = [], []
        for _ in range(2):
            fw_lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, state_is_tuple=True)
            # 添加dropout.为了防止过拟合，在它的隐层添加了 dropout 正则
            fw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(fw_lstm_cell, output_keep_prob=self.dropout)
            fw_lstms.append(fw_lstm_cell)

            bw_lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_dim, state_is_tuple=True)
            bw_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(bw_lstm_cell, output_keep_prob=self.dropout)
            bw_lstms.append(bw_lstm_cell)
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            fw_lstms,
            bw_lstms,
            model_inputs,
            sequence_length=lengths,
            dtype=tf.float32)
        return outputs

    def project_layer_bilstm(self, lstm_outputs, num):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"):
            with tf.variable_scope("attention"):
                if self.use_attention:
                    attention_outputs = attention_layer(lstm_outputs,
                                                        lstm_outputs,
                                                        self.mask_inputs,
                                                        self.num_attention_heads,
                                                        self.size_per_head)
                else:
                    attention_outputs = lstm_outputs
            with tf.variable_scope("hidden"):
                if self.use_attention:
                    w_shape = [self.num_attention_heads * self.size_per_head, self.lstm_dim]
                    output_shape = [-1,  self.num_attention_heads * self.size_per_head]
                else:
                    w_shape = [self.lstm_dim * num, self.lstm_dim]
                    output_shape = [-1, self.lstm_dim * num]
                W = tf.get_variable("W",
                                    shape=w_shape,
                                    dtype=tf.float32,
                                    initializer=self.initializer)
                b = tf.get_variable("b",
                                    shape=[self.lstm_dim],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(attention_outputs, shape=output_shape)
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W",
                                    shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32,
                                    initializer=self.initializer)
                b = tf.get_variable("b",
                                    shape=[self.num_tags],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def inference(self, embedding):
        model_inputs = tf.nn.dropout(embedding, self.dropout)
        if self.layer_type == 'single':
            model_outputs = self.single_biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)
            logits = self.project_layer_bilstm(model_outputs, 2)
        elif self.layer_type == 'stack':
            model_outputs = self.stack_biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)
            logits = self.project_layer_bilstm(model_outputs, 2)
        else:
            model_outputs = self.concat_biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)
            logits = self.project_layer_bilstm(model_outputs, 4)
        return logits

    def loss_layer(self, project_logits, lengths):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat([small * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                                      tf.zeros(shape=[self.batch_size, 1, 1])],
                                     axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat([tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable("transitions",
                                         shape=[self.num_tags + 1, self.num_tags + 1],
                                         initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(inputs=logits,
                                                            tag_indices=targets,
                                                            transition_params=self.trans,
                                                            sequence_lengths=lengths + 1)
            loss = tf.reduce_mean(-log_likelihood)
            return loss

    def create_feed_dict(self, is_train, batch):
        str_input, masks, tags = batch
        token_ids = self.batcher.batch_sentences(str_input)
        feed_dict = {
            self.context_token_ids: np.asarray(token_ids),
            self.mask_inputs: np.asarray(masks),
            self.dropout: 1.0
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.input_dropout
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss,  _ = sess.run([self.global_step, self.loss, self.train_op], feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager):
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = [[int(x)] for x in tags[i][:lengths[i]]]
                pred = [[int(x)] for x in batch_paths[i][:lengths[i]]]
                for char, gold, pred in zip(string, gold, pred):
                    result.append([char, gold, pred])
                results.append(result)
        return results


def create_model(session, Model_class, path, config, logger):
    model = Model_class(config)
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        try:
            logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        except:
            logger.info("Reading model parameters Failed. Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


