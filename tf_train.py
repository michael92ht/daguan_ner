# encoding: utf-8

import time
import tensorflow as tf
from model.keras_model import *
from data_utils import *
from model.tf_model import *
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import warnings
np.random.seed(0)
warnings.filterwarnings('ignore')


flags = tf.app.flags
flags.DEFINE_boolean("train",       True,      "Whether train the model")
flags.DEFINE_boolean("clean",       True,      "Whether clean")
flags.DEFINE_string("model_type",  r'tf', "")
flags.DEFINE_string("config_file",  r'D:\work\daguan\data\config', "")
flags.DEFINE_string("log_file",  r'D:\work\daguan\data\log.txt', "")

# configurations for the model
flags.DEFINE_integer("lstm_dim",    200,        "Num of hidden units in LSTM")
flags.DEFINE_string("layer_type",    'concat',        "concat or stack")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_boolean("attention",       True,      "Whether use attention")
flags.DEFINE_float("lr",            0.0003,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_integer("max_epoch",   120,        "maximum training epochs")
flags.DEFINE_integer("models_count",    6,        "Num of models_count")
flags.DEFINE_integer("train_beg",    0,        "Num of models_count")
flags.DEFINE_integer("num_attention_heads", 4,  "")
flags.DEFINE_integer("size_per_head", 128,  "")

flags.DEFINE_string("vocab_file",  r'D:\work\daguan\data\elmo_data\vocab.txt',"")
flags.DEFINE_string("options_file",  r'D:\work\daguan\data\elmo_model\options.json',"")
flags.DEFINE_string("weight_file",  r'D:\work\daguan\data\elmo_model\weights.hdf5',"")
flags.DEFINE_string("token_embedding_file",  r'D:\work\daguan\data\elmo_model\vocab_embedding.hdf5',"")
flags.DEFINE_string("model_hub",  r'D:\work\daguan\data\hub', "")
flags.DEFINE_string("result_hub",  r'D:\work\daguan\data\result', "")
flags.DEFINE_string("dataset",  r'D:\work\daguan\data\anns\raw_dataset.pkl', "")
flags.DEFINE_string("test_file",  r'D:\work\daguan\data\test.txt', "")
FLAGS = flags.FLAGS

LOG = get_logger(FLAGS.log_file)


# config for the model
def config_model():
    config = {}
    config["model_type"] = FLAGS.model_type
    config["lstm_dim"] = FLAGS.lstm_dim
    config["dropout"] = FLAGS.dropout
    config["layer_type"] = FLAGS.layer_type
    config["lr"] = FLAGS.lr
    config['vocab_file'] = FLAGS.vocab_file
    config['options_file'] = FLAGS.options_file
    config['weight_file'] = FLAGS.weight_file
    config['token_embedding_file'] = FLAGS.token_embedding_file
    config['size_per_head'] = FLAGS.size_per_head
    config['num_attention_heads'] = FLAGS.num_attention_heads
    config['attention'] = FLAGS.attention
    return config


TAGS_NUM = 7
BATCH_SIZE = 128
STEP_CHECK = 30


def get_train_val_data(train_data, tags, split_index, model_count):
    # split data into train/vali set
    idx_val = split_index[model_count]
    idx_train = []
    for i in range(len(split_index)):
        if i != model_count:
            idx_train.extend(list(split_index[i]))

    train_sentences = train_data[idx_train]
    train_tags = tags[idx_train]
    trains = [(train_sentences[x], train_tags[x]) for x in range(len(train_tags))]
    train_manager = BatchManager(trains, False)

    val_sentences = train_data[idx_val]
    val_tags = tags[idx_val]
    vals = [(val_sentences[x], val_tags[x]) for x in range(len(val_tags))]
    val_manager = BatchManager(vals, False)
    return train_manager, val_manager


def evaluate_val(sess, model, data):
    LOG.info("evaluate............")
    results = model.evaluate(sess, data)
    y = [x[1][0] for result in results for x in result]
    pred = [x[2][0] for result in results for x in result]
    report = classification_report(y_pred=pred, y_true=y)
    LOG.info('\n' + report)
    f1 = f1_score(y, pred, average='macro')
    recall = recall_score(y, pred, average='macro')
    precision = precision_score(y, pred, average='macro')
    LOG.info("step: {} - val_precision: {} - val_recall: {} - val_f1: {}".format(model.global_step.eval(), precision,  recall, f1))
    best_test_f1 = model.best_dev_f1.eval()
    if f1 > best_test_f1:
        tf.assign(model.best_dev_f1, f1).eval()
        LOG.info("new best dev f1 score:{:>.6f}".format(f1))
    return f1 > best_test_f1, f1


def run():
    if not FLAGS.clean and os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model()
        save_config(config, FLAGS.config_file)
    with open(FLAGS.dataset, 'rb') as f:
        data, tags, test_data, split_index = pickle.load(f)
    raw_test_data = read_corpus_file(FLAGS.test_file)
    LOG.info('sentences: {}'.format(len(data)))
    LOG.info('tags: {}'.format(len(tags)))
    LOG.info('test data: {}'.format(len(test_data)))
    LOG.info('raw test data: {}'.format(len(raw_test_data)))
    # split dataset
    input_data = (data, tags, test_data)
    encoder_type = FLAGS.model_type
    train_data, tags, test_data = input_data
    test_manager = BatchManager(test_data, True)
    best_val_score = {}
    model_save_folder = os.path.join(FLAGS.model_hub, encoder_type)
    result_save_folder = os.path.join(FLAGS.result_hub, encoder_type)
    for folder in [model_save_folder, result_save_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    for model_count in range(FLAGS.train_beg, len(split_index)):
        LOG.info("MODEL: {}".format(model_count))
        train_manager, val_manager = get_train_val_data(train_data, tags, split_index, model_count)
        ckpt_path = os.path.join(model_save_folder, str(model_count).zfill(3))
        result_save_path = os.path.join(result_save_folder, str(model_count).zfill(3) + '.npy')
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
        steps_per_epoch = train_manager.len_data
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = create_model(sess, Model, ckpt_path, config, LOG)
            loss = []
            for i in range(FLAGS.max_epoch):
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss = model.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if step % STEP_CHECK == 0:
                        iteration = step // steps_per_epoch + 1
                        LOG.info("iteration:{} step:{}/{}, NER loss:{:>9.6f}".format(iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []

                best, best_f1 = evaluate_val(sess, model, val_manager)
                if best:
                    save_model(sess, model, ckpt_path)
                    best_val_score[model_count] = best_f1
            # save epoch test result
            preds = model.evaluate(sess, test_manager)
            test_preds = []
            for pred in preds:
                test_preds.append([(x[0], x[2][0]) for x in pred])
            np.save(result_save_path, np.array(test_preds))

    for index, f1 in best_val_score.items():
        LOG.info(str(index) + ':\t' + str(f1))
    result_path = os.path.join(FLAGS.result_hub, encoder_type + '_result.txt')
    write_tf_result(raw_test_data, result_save_folder, result_path)


def test():
    config = load_config(FLAGS.config_file)
    raw_test_data = read_corpus_file(FLAGS.test_file)
    test_manager = BatchManager(raw_test_data, is_test=True)
    model_save_folder = os.path.join(FLAGS.model_hub, FLAGS.model_type)
    result_save_folder = os.path.join(FLAGS.result_hub, 'test_' + FLAGS.model_type)
    for folder in [model_save_folder, result_save_folder]:
        if not os.path.exists(folder):
            os.mkdir(folder)

    for model_count in range(FLAGS.models_count):
        LOG.info("MODEL: {}".format(model_count))
        ckpt_path = os.path.join(model_save_folder, str(model_count).zfill(3))
        result_save_path = os.path.join(result_save_folder, str(model_count).zfill(3) + '.npy')
        if not os.path.exists(ckpt_path):
            continue
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = create_model(sess, Model, ckpt_path, config, LOG)
            preds = model.evaluate(sess, test_manager)
            test_preds = []
            for pred in preds:
                test_preds.append([(x[0], x[2][0]) for x in pred])
            np.save(result_save_path, np.array(test_preds))
            LOG.info("Dump result to {}".format(result_save_path))
    result_path = os.path.join(result_save_folder, FLAGS.model_type + '_result.txt')
    write_tf_result(raw_test_data, result_save_folder, result_path)
    LOG.info("Write results to {}".format(result_path))


def main(_):
    if FLAGS.train:
        start = time.time()
        run()
        end = time.time()
        LOG.info('Training time {0:.3f} 分钟'.format((end - start) / 60))
    else:
        test()


if __name__ == "__main__":
    tf.app.run(main)

