# encoding: utf-8

import os
import json
import codecs
import pickle
import logging
import numpy as np
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def save_model(sess, model, path):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)


def save_config(config, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        return json.load(f)


MAX_SEN_LEN = 200
LABELS_2_ID = {
    'O': 0,
    'B-A': 1,
    'I-A': 2,
    'B-B': 3,
    'I-B': 4,
    'B-C': 5,
    'I-C': 6
}
ID_2_LABELS = {v: k for k, v in LABELS_2_ID.items()}


def read_corpus_file(path):
    lines = []
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f.read().split('\n'):
            words = [x.strip() for x in line.split('_') if len(x.strip())]
            lines.append(words)
    return lines


def convert_to_w2v_file(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        return f.read().replace('_', ' ')


def build_w2v_corpus():
    folder = '../data/corpus'
    texts = []
    for p in os.listdir(folder):
        pf = os.path.join(folder, p)
        with codecs.open(pf, 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)
    with codecs.open(os.path.join(folder, 'total.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(texts))
    print('bulit done!')


def get_word_dict(path):
    word_tags = []
    with codecs.open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines:
            parts = line.split('  ')
            for part in parts:
                text, tag = part.split('/')
                # part_words = text.split('_')
                if tag != 'o':
                    word_tags.append((text, tag))
    mult_tags_word = set()
    temp = {}
    for w, t in word_tags:
        if w in temp.keys() and t != temp[w]:
            mult_tags_word.add(w)
        elif not w in temp.keys():
            temp[w] = t

    counter = Counter(word_tags)
    word_tag_counter =sorted([(k, v) for k, v in counter.items() if k[0] not in mult_tags_word and v > 10 * (7 - len(k[0].split('_')))], key=lambda x: -x[1])
    for word_tag, counter in word_tag_counter:
        # if word_tag[0] == '2797_18850_537_14499_2242_4246_17592':
        print(word_tag, counter)
    print('-----------------------')
    words = sorted([x[0][0] for x in word_tag_counter])
    # for w in words:
    #     #     print(w)

    for w in Counter(words).most_common(20):
        print(w)


def read_ann_file(path):
    def process_ann_line(line):
        parts = line.split('  ')
        tokens, tags = [], []
        for part in parts:
            text,  tag = part.split('/')
            part_words = text.split('_')
            part_tags = []
            if tag == 'o':
                part_tags = ['O' for _ in range(len(part_words))]
            else:
                current_tag = tag.upper()
                part_tags.append('B-' + current_tag)
                for _ in range(len(part_words) - 1):
                    part_tags.append('I-' + current_tag)
            tokens.extend(part_words)
            tags.extend(part_tags)
        assert len(tokens) == len(tags)
        return tokens, tags
    sentences, ner_tags = [], []
    with codecs.open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines:
            tokens, tags = process_ann_line(line)
            sentences.append(tokens)
            ner_tags.append(tags)
    return sentences, ner_tags


def dump_ann_file():
    ann_file = '../data/train.txt'
    sentences, ner_tags = read_ann_file(ann_file)
    with open('../data/anns/ann_dataset.pkl', 'wb') as f:
        pickle.dump((sentences, ner_tags), f)


# elmo
def prepare_elmo_dataset(w2v_file,  vocab_path, corpus_path, out_folder):
    with codecs.open(w2v_file, 'r', encoding='utf-8') as f:
        words = [x.split(' ')[0] for x in f.read().split('\n') if x.strip()]
    words.pop(0)
    begs = ['<S>', '</S>', '<UNK>']
    with codecs.open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(begs + words))
    print('convert to elmo vocab file ', vocab_path)

    with codecs.open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = f.read().split('\n')
    length = len(corpus)
    step = length // 30
    for index in range(30):
        beg = index * step
        file = os.path.join(out_folder, 'elmo_corpus_' + str(index).zfill(3) + '.txt')
        with codecs.open(file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(corpus[beg: beg + step]))
        print('Write to ', file)


def load_w2v_file(w2v_file):
    embeddings = []
    word_2_id = {}
    with codecs.open(w2v_file, 'r', encoding='utf-8') as f:
        lines = [x for x in f.read().split('\n') if x.strip()]
    first_line = [int(x.strip()) for x in lines[0].split(' ')]
    count, dim = first_line[0], first_line[1]
    zeros_emb = np.zeros(dim)
    embeddings.append(zeros_emb)
    embeddings.append(zeros_emb)
    word_2_id['<pad>'] = 0
    word_2_id['<unk>'] = 1
    for index, line in enumerate(lines[1:]):
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        assert coefs.shape == (dim, ), (coefs.shape, index)
        embeddings.append(coefs)
        word_2_id[word] = index + 2
    print('Found %s word vectors.' % len(embeddings))
    return embeddings, word_2_id


def get_cut_indexs(tags):
    length = len(tags)
    indexs = []
    beg, end = 0, MAX_SEN_LEN
    while end < length:
        while sum(tags[end - 5: end + 5]) != 0:
            end -= 1
        if end <= beg + MAX_SEN_LEN // 2:
            end = beg + MAX_SEN_LEN
        indexs.append((beg, end))
        beg = end
        end = beg + MAX_SEN_LEN
    indexs.append((beg, length))

    for index in range(len(indexs) - 1):
        current, next_one = indexs[index], indexs[index + 1]
        assert current[1] == next_one[0]
    return indexs


def cut_and_pad(sentence, tags):
    processed_sens, processed_tags = [], []
    length = len(sentence)
    if length <= MAX_SEN_LEN:
        processed_sens.append(sentence)
        processed_tags.append(tags)
    else:
        indexs = get_cut_indexs(tags)
        for (beg, end) in indexs:
            sen = sentence[beg: end]
            tag = tags[beg: end]
            processed_sens.append(sen)
            processed_tags.append(tag)
    return processed_sens, processed_tags


def load_embedding_file(w2v_file):
    embeddings, word_2_id = load_w2v_file(w2v_file)
    embeddings = np.array(embeddings)
    return embeddings, word_2_id


def load_train_file(train_file, word_2_id):
    sentences, ner_tags = read_ann_file(train_file)
    tokenizer_sentences, tags = [], []
    for sen in sentences:
        tokenizer_sentences.append([word_2_id.get(word, 1) for word in sen])  # 1 for unk
    for tag_seq in ner_tags:
        tags.append([LABELS_2_ID[label] for label in tag_seq])
    total_sens, total_tags = [], []
    for index in range(len(tags)):
        sens, sen_tags = tokenizer_sentences[index], tags[index]
        processed_sens, processed_tags = cut_and_pad(sens, sen_tags)
        total_sens.extend(processed_sens)
        total_tags.extend(processed_tags)
    tokenizer_sentences = np.array(pad_sequences(total_sens, MAX_SEN_LEN, padding='post', value=0))
    tags = np.array(pad_sequences(total_tags, MAX_SEN_LEN, padding='post', value=0))
    assert tokenizer_sentences.shape == tags.shape, (tokenizer_sentences.shape, tags.shape)
    return tokenizer_sentences, tags


def load_test_file(test_file, word_2_id):
    test_map = defaultdict(list)
    test_mask = {}
    test_inputs = []
    raw_test_data = read_corpus_file(test_file)
    for index, line in enumerate(raw_test_data):
        line = [word_2_id.get(word, 1) for word in line]
        length = len(line)
        test_mask[index] = length
        if length <= MAX_SEN_LEN:
            test_inputs.append(line)
            test_map[index].append(len(test_inputs) - 1)
        else:
            beg, end = 0, MAX_SEN_LEN
            while end <= length:
                test_inputs.append(line[beg: end])
                test_map[index].append(len(test_inputs) - 1)
                beg, end = end,  end + MAX_SEN_LEN
            if end > length:
                test_inputs.append(line[beg: end])
                test_map[index].append(len(test_inputs) - 1)
    test_data = np.array(pad_sequences(test_inputs, MAX_SEN_LEN, padding='post', value=0))
    return test_data, test_map, test_mask, raw_test_data


def build_dataset(w2v_file, train_file, test_file, dump_file):
    embeddings, word_2_id = load_embedding_file(w2v_file)
    train_data, train_tags = load_train_file(train_file, word_2_id)
    test_data, test_data_map, test_mask, raw_test_data = load_test_file(test_file, word_2_id)
    with open(dump_file, 'wb') as f:
        pickle.dump((train_data, train_tags, test_data, test_data_map, test_mask, raw_test_data, embeddings, word_2_id),
                    f,  protocol=2)
    print('builded dataset........')


def vote(results):
    res_dict = {}
    for d in results:
        if d in res_dict.keys():
            res_dict[d] += 1
        else:
            res_dict[d] = 1
    most = sorted([(k, v) for k, v in res_dict.items()], key=lambda x: -x[1])[0]
    return most[0]
    # if most[1] > len(results) // 2:
    #     return most[0]
    # else:
    #     return 0


def merge_results(folder, test_dataset):
    test_data, test_data_map, test_mask = test_dataset
    preds = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if path.endswith('npy'):
            preds.append(np.load(path))

    merged = []
    for index in range(len(test_data)):
        tmp = []
        for j in range(MAX_SEN_LEN):
            tmp.append([preds[p][index][j] for p in range(len(preds))])
        merged.append(tmp)
    test_results = {}
    for test_index, sub_indexs in test_data_map.items():
        if len(sub_indexs) == 1:
            merged_result = merged[sub_indexs[0]]
        else:
            merged_result = []
            beg = 0
            merged_result.extend(merged[sub_indexs[beg]])
            beg += 1
            while beg < len(sub_indexs):
                merged_result.extend(merged[sub_indexs[beg]])
                beg += 1
        tmp = [vote(m) for m in merged_result]
        test_results[test_index] = tmp[:test_mask[test_index]]
    return test_results


def write_results(test_results, raw_test_data, result_path='test_result.txt'):
    bad_cases_count = 0
    results = []
    for index, words in enumerate(raw_test_data):
        tags = [ID_2_LABELS[x] for x in test_results[index]]
        length = len(tags)
        beg = 0
        temp = ''
        while beg < length:
            if beg < length and tags[beg] == 'O':
                while beg < length and tags[beg] == 'O':
                    temp += words[beg] + '_'
                    beg += 1
                if beg == length:
                    temp = temp[:-1] + '/o'
                elif beg > 0:
                    temp = temp[:-1] + '/o  '

            if beg < length and tags[beg][0] in {'B', 'I'}:
                if tags[beg][0] == 'I':
                    bad_cases_count += 1
                    print('==================bad predict================')
                temp += words[beg] + '_'
                beg += 1
                while beg < length and tags[beg][0] == 'I':
                    temp += words[beg] + '_'
                    beg += 1
                if beg == length:
                    temp = temp[:-1] + '/' + tags[beg - 1][2].lower()
                    beg += 100
                else:
                    temp = temp[:-1] + '/' + tags[beg - 1][2].lower() + '  '
        # print(temp)
        h = [x.strip() for y in temp.split('  ') for x in y[:-2].split('_')]
        assert h == words
        results.append(temp)

    with codecs.open(result_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results) + '\n')
    print("Write results to ", result_path)
    print('Bad case count ', bad_cases_count)


def get_split_indexs(length, split_num):
    indexs = np.arange(length)
    np.random.shuffle(indexs)
    split_index = {}
    step = length // split_num
    for i in range(split_num):
        beg = i * step
        split_index[i] = indexs[beg: beg + step]
    return split_index


def decode(test_preds):
    out = []
    for pred in test_preds:
        out_i = [np.argmax(p) for p in pred]
        out.append(out_i)
    return np.array(out)


def build_raw_dataset(train_file, test_file, dump_file, split_num=6):
    sentences, ner_tags = read_ann_file(train_file)
    tags = []
    for ts in ner_tags:
        tags.append([LABELS_2_ID[x] for x in ts])
    raw_test_data = read_corpus_file(test_file)
    length = len(sentences)
    split_index = get_split_indexs(length, split_num)
    with open(dump_file, 'wb') as f:
        pickle.dump((np.array(sentences), np.array(tags), np.array(raw_test_data), split_index), f, protocol=2)
    print('builded dataset........')


def write_tf_result(raw_test_data, folder, result_path):
    preds = defaultdict(list)
    for file in os.listdir(folder):
        if not file.endswith('npy'):
            continue
        path = os.path.join(folder, file)
        tmp = np.load(path, allow_pickle=True)
        for t in tmp:
            preds['_'.join([x[0] for x in t])].append([x[1] for x in t])
    voted_preds = {}
    for key, tags in preds.items():
        voted_preds[key] = [vote([tag[index] for tag in tags]) for index in range(len(tags[0]))]
    test_results = []
    for raw_data in raw_test_data:
        line = '_'.join(raw_data)
        assert line in voted_preds.keys(), (line)
        test_results.append(voted_preds[line])
    write_results(test_results, raw_test_data, result_path)


if __name__ == '__main__':
    # corpus_file = '../data/test.txt'
    # converted = convert_to_w2v_file(corpus_file)
    # with codecs.open('../data/corpus/test.txt', 'w', encoding='utf-8') as f:
    #     f.write(converted)
    # lines = read_corpus_file(corpus_file)

    # with codecs.open('../data/corpus/train.txt', 'w', encoding='utf-8') as f:
    #     f.write('\n'.join([' '.join(x) for x in sentences]))
    # build_dataset('../data/w2v/wv.vector')
    # get_word_dict('../data/train.txt')
    # test_file = '../data/test.txt'
    # train_file = '../data/train.txt'
    # w2v_file = '../data/w2v/wv.vector'
    # dump_file = '../data/anns/dataset.pkl'
    # build_dataset(w2v_file, train_file, test_file, dump_file)
    # h = load_w2v_file(w2v_file)
    # vocab_path = '../data/elmo_data/vocab.txt'
    # corpus_path = '../data/corpus/total.txt'
    # out_folder = '../data/elmo_data/'
    # prepare_elmo_dataset(w2v_file, vocab_path, corpus_path, out_folder)
    test_file = 'data/test.txt'
    train_file = 'data/train.txt'
    dump_file = 'data/anns/raw_dataset.pkl'
    build_raw_dataset(train_file, test_file, dump_file)
    # folder = r'D:\work\daguan\data\result\tf'
    # write_tf_result(folder, result_path='test.txt')





    pass