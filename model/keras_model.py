# encoding: utf-8


import keras.callbacks as kcallbacks
from .base import *
np.random.seed(1)

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


LSTM_NUM = 256
DROP_OUT = 0.5
BATCH_SIZE = 128
EPOCHS = 100
SPLIT_NUM = 10

MAX_NB_WORDS = 10000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 26


class CustomMetrics(kcallbacks.Callback):
    def __init__(self, filepath):
        super(CustomMetrics).__init__()
        self.file_path = filepath

    def on_train_begin(self, logs={}):
        self.best_val_f1 = 0
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def flatten_result(self, result):
        out = []
        for pred_i in result:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p)
                out_i.append(p_i)
            out.extend(out_i)
        return out

    def on_epoch_end(self, epoch, logs={}):
        val_predict = self.model.predict(self.validation_data[0])
        pred = self.flatten_result(val_predict)
        y = self.flatten_result(self.validation_data[1])

        report = classification_report(y_pred=pred, y_true=y)
        print(report)

        _val_f1 = f1_score(y, pred, average='macro')
        _val_recall = recall_score(y, pred, average='macro')
        _val_precision = precision_score(y, pred, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("epoch: {} - val_precision: {} - val_recall: {} - val_f1: {}".format(epoch + 1, _val_precision, _val_recall, _val_f1))
        if _val_f1 > self.best_val_f1:
            self.model.save(self.file_path, overwrite=True)
            self.best_val_f1 = _val_f1
            print("new best f1: {}".format(self.best_val_f1))
        return


def dgcnn_encoder(q_embed):
    # t = Conv1D(128, 3, activation='relu', kernel_initializer='glorot_uniform', padding='same')(q)
    t = dilated_gated_conv1d(q_embed, 1)
    t = dilated_gated_conv1d(t, 3)
    t = dilated_gated_conv1d(t, 5)

    t = dilated_gated_conv1d(t, 1)
    t = dilated_gated_conv1d(t, 3)
    t = dilated_gated_conv1d(t, 5)

    #     t = dilated_gated_conv1d(t, 1)
    #     t = dilated_gated_conv1d(q, 3)
    #     t = dilated_gated_conv1d(t, 5)

    #     t = dilated_gated_conv1d(t, 1)
    #     t = dilated_gated_conv1d(q, 3)
    #     t = dilated_gated_conv1d(t, 5)
    #     t = dilated_gated_conv1d(t, 3)
    #     t = dilated_gated_conv1d(t, 1)
    #     t = dilated_gated_conv1d(t, 2)
    #     t = dilated_gated_conv1d(t, 5)
    #     t = dilated_gated_conv1d(t, 1)
    #     t = dilated_gated_conv1d(t, 2)
    #     t = dilated_gated_conv1d(t, 5)
    t = dilated_gated_conv1d(t, 1)
    t = dilated_gated_conv1d(t, 1)
    q = dilated_gated_conv1d(t, 1)
    q = Dropout(DROP_OUT)(q)
    q = BatchNormalization()(q)
    #     q = Flatten()(t)
    q = Attention(8, 128)([q, q, q])  # output: [batch_size, time_step, nb_head*size_per_head]
    q = Dropout(DROP_OUT)(q)
    # avg_pool = GlobalAveragePooling1D()(q)
    # max_pool = GlobalMaxPooling1D()(q)
    # q = concatenate([avg_pool, max_pool])
    #     q = Dropout(0.5)(q)
    return q


def attention_encoder(q_embed):
    #     cnn_1 = Conv1D(128, 2, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    #     cnn_2 = Conv1D(128, 3, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    #     cnn_3 = Conv1D(128, 5, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    #     q = concatenate([cnn_1(q_embed), cnn_2(q_embed), cnn_3(q_embed)])
    #     shared_lstm = CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform', return_sequences=True)
    #     q = shared_lstm(q_embed)
    shared_lstm_1 = Bidirectional(CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform', return_sequences=True))
    shared_lstm_2 = Bidirectional(CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform', return_sequences=True))

    q1 = shared_lstm_1(q_embed)
    # q1 = Dropout(DROP_OUT)(q1)
    q1 = BatchNormalization()(q1)
    q2 = shared_lstm_2(q1)
    # q2 = Dropout(DROP_OUT)(q2)
    q2 = BatchNormalization()(q2)
    q = concatenate([q_embed, q1, q2])
    #     q = Position_Embedding(20, mode='concat')(q)
    q = Attention(8, 64)([q, q, q])  # output: [batch_size, time_step, nb_head*size_per_head]
    # q = GlobalAveragePooling1D()(q)
    q = Dropout(DROP_OUT)(q)
    q = BatchNormalization()(q)
    q = TimeDistributed(Dense(LSTM_NUM, activation="relu"))(q)            # a dense layer as suggested by neuralNer
    return q


def lstm_encoder(q_embed):
    shared_lstm = CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform')
    q = shared_lstm(q_embed)
    q = Dropout(DROP_OUT)(q)
    q = BatchNormalization()(q)
    return q


def two_lstm_encoder(q_embed):
    shared_lstm_1 = CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform', return_sequences=True)
    shared_lstm_2 = CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform')

    q = shared_lstm_1(q_embed)
    q = Dropout(DROP_OUT)(q)
    q = BatchNormalization()(q)
    q = shared_lstm_2(q)
    return q


def bilstm_encoder(q_embed):
    shared_lstm_1 = Bidirectional(CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform'))
    q = shared_lstm_1(q_embed)
    q = Dropout(DROP_OUT)(q)
    q = BatchNormalization()(q)
    return q


def two_bilstm_encoder(q_embed):
    shared_lstm_1 = Bidirectional(
        CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform', return_sequences=True))
    shared_lstm_2 = Bidirectional(CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform'))
    q = shared_lstm_1(q_embed)
    q = Dropout(DROP_OUT)(q)
    q = BatchNormalization()(q)
    q = shared_lstm_2(q)
    return q


def cnn_encoder(q_embed):
    cnn_1 = Conv1D(128, 2, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    cnn_2 = Conv1D(128, 3, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    cnn_3 = Conv1D(128, 5, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    # pool = MaxPooling1D(5)
    q = concatenate([cnn_1(q_embed), cnn_2(q_embed), cnn_3(q_embed)])
    q = Dropout(DROP_OUT)(q)
    # q = pool(q)
    return q


def lstm_cnn_encoder(q_embed):
    shared_lstm_1 = CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform', return_sequences=True)
    cnn_1 = Conv1D(128, 2, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    cnn_2 = Conv1D(128, 3, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    cnn_3 = Conv1D(128, 5, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    pool = MaxPooling1D(5)
    q = shared_lstm_1(q_embed)
    q = Dropout(DROP_OUT)(q)
    q = BatchNormalization()(q)
    q = concatenate([cnn_1(q), cnn_2(q), cnn_3(q)])
    q = Dropout(DROP_OUT)(q)
    q = pool(q)
    q = Flatten()(q)
    return q


def bilstm_cnn_encoder(q_embed):
    shared_lstm_1 = Bidirectional(
        CuDNNLSTM(LSTM_NUM, kernel_initializer='glorot_uniform', return_sequences=True))
    cnn_1 = Conv1D(128, 2, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    cnn_2 = Conv1D(128, 3, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    cnn_3 = Conv1D(128, 5, activation='relu', kernel_initializer='glorot_uniform', padding='same')
    pool = MaxPooling1D(5)
    q = shared_lstm_1(q_embed)
    q = Dropout(DROP_OUT)(q)
    q = BatchNormalization()(q)
    q = concatenate([cnn_1(q), cnn_2(q), cnn_3(q)])
    q = Dropout(DROP_OUT)(q)
    q = pool(q)
    q = Flatten()(q)
    return q

