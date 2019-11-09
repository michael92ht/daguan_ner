# encoding: utf-8

from keras.models import Model
from keras_contrib.layers import CRF
from keras.utils import to_categorical
from model.keras_model import *
from utils.data_utils import *
import warnings
np.random.seed(0)
warnings.filterwarnings('ignore')

SPLIT_NUM = 6
MAX_SEQUENCE_LENGTH = 200
TAGS_NUM = 7
EMBEDDING_DIM = 300
DROP_OUT = 0.5
LSTM_NUM = 200
BATCH_SIZE = 128
# 128
EPOCHS = 300
MASK_ZERO = True

encoders_dict = {
    'attention': attention_encoder,
    'cnn': cnn_encoder,
    'lstm': lstm_encoder,
    'bilstm': bilstm_encoder,
    '2lstm': two_lstm_encoder,
    '2bilstm': two_bilstm_encoder,
    'lstmcnn': lstm_cnn_encoder,
    'bilstmcnn': bilstm_cnn_encoder,
    'dgcnn': dgcnn_encoder
}

def run(input_data, split_index, encoder_type='cnn'):
    train_data, tags, test_data, test_data_map, test_mask, raw_test_data, embed_matrix, word_2_id = input_data
    best_vali_score = {}
    model_save_folder = 'data/hub/' + encoder_type
    test_save_folder = 'data/result/' + encoder_type
    for folder in [model_save_folder, test_save_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    for model_count in range(SPLIT_NUM):
        print("MODEL:", model_count)

        # split data into train/vali set
        idx_val = split_index[model_count]
        idx_train = []
        for i in range(SPLIT_NUM):
            if i != model_count:
                idx_train.extend(list(split_index[i]))

        train_sentences = train_data[idx_train]
        train_tags = tags[idx_train]
        train_tags = np.array([to_categorical(i, num_classes=TAGS_NUM) for i in train_tags])

        val_sentences = train_data[idx_val]
        val_tags = tags[idx_val]
        val_tags = np.array([to_categorical(i, num_classes=TAGS_NUM) for i in val_tags])

        input = Input(shape=(MAX_SEN_LEN,))
        # model = Masking(mask_value=0, input_shape=(MAX_SEN_LEN,))(input)
        if 'cnn' in encoder_type or 'attention' in encoder_type:
            mask_zero = False
        else:
            mask_zero = True
        embed = Embedding(embed_matrix.shape[0],
                          weights=[embed_matrix],
                          trainable=True,
                          output_dim=EMBEDDING_DIM,
                          input_length=MAX_SEN_LEN,
                          mask_zero=mask_zero)(input)
        # q = cnn_encoder(embed)
        # q = two_lstm_encoder(embed)
        # q0 = embed
        q0 = Dropout(DROP_OUT)(embed)
        # q = encoders_dict[encoder_type](q0)
        q1 = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.2))(q0)        # variational biLSTM
        q2 = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.2))(q1)        # variational biLSTM
        q = concatenate([q1, q2])
        q = TimeDistributed(Dense(100, activation="relu"))(q)            # a dense layer as suggested by neuralNer
        crf = CRF(TAGS_NUM)  # CRF layer
        out = crf(q)

        model = Model(input, out)
        model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        best_weights_filepath = os.path.join(model_save_folder, str(model_count).zfill(3) + '.hdf5')
        custom_metrics = CustomMetrics(best_weights_filepath)
        # define save model
        earlyStopping = kcallbacks.EarlyStopping(monitor='crf_viterbi_accuracy',
                                                 patience=15,
                                                 verbose=1,
                                                 mode='auto')
        # saveBestModel = kcallbacks.ModelCheckpoint(
        #     best_weights_filepath,
        #     monitor='crf_viterbi_accuracy',
        #     verbose=1,
        #     save_best_only=True,
        #     mode='auto')

        hist = model.fit(
            train_sentences, train_tags,
            validation_data=(val_sentences, val_tags),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=[earlyStopping, custom_metrics],
            verbose=1)

        model.load_weights(best_weights_filepath)
        print(model_count, "validation loss:", min(hist.history["val_loss"]))
        best_vali_score[model_count] = min(hist.history["val_loss"])

        # predict on the test set
        test_preds = model.predict(test_data, batch_size=128, verbose=1)
        test_preds = decode(test_preds)
        save_path = os.path.join(test_save_folder, str(model_count).zfill(3) + '.npy')
        np.save(save_path, test_preds)
        print("Test preds saved: ", model_count)

    for index, loss in best_vali_score.items():
        print(encoder_type, index, loss)

    result_path = os.path.join('data/result', encoder_type + '_result.txt')
    test_dataset = (test_data, test_data_map, test_mask)
    test_results = merge_results(test_save_folder, test_dataset)
    write_results(test_results, raw_test_data, result_path)


if __name__ == '__main__':
    # load dataset
    with open('data/anns/dataset.pkl', 'rb') as f:
        data, tags, test_data, test_data_map, test_mask, raw_test_data, embeddings, word_2_id = pickle.load(f)
    print('sentences: ', data.shape)
    print('tags: ', tags.shape)
    print('embed_matrix: ', embeddings.shape)

    # split dataset
    length = data.shape[0]
    split_index = get_split_indexs(length, SPLIT_NUM)
    encoder_type = '2bi'

    # import time
    # start = time.time()
    # input_data = (data, tags, test_data, test_data_map, test_mask, raw_test_data, embeddings, word_2_id)
    # run(input_data, split_index, encoder_type)
    # end = time.time()
    # print('Training time {0:.3f} 分钟'.format((end - start) / 60))

    result_path = os.path.join('data/result',  encoder_type + '_result.txt')
    test_dataset = (test_data, test_data_map, test_mask)
    test_results = merge_results('data/result/' + encoder_type, test_dataset)
    write_results(test_results, raw_test_data, result_path)

