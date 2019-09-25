# coding=utf-8
import nltk
import pickle


def chinese_process(filein, fileout):
    with open(filein, 'r') as infile:
        with open(fileout, 'w') as outfile:
            for line in infile:
                output = list()
                line = nltk.word_tokenize(line)[0]
                for char in line:
                    output.append(char)
                    output.append(' ')
                output.append('\n')
                output = ''.join(output)
                outfile.write(output)


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)
    for sentence in tokens:
        index = 0
        for word in sentence:
            code_str += (str(dictionary[word]) + ' ')
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str


def code_to_text(codes, dictionary):
    paras = ""
    eof_code = len(dictionary)
    for line in codes:
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                continue
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


def get_tokenlized(file):
    tokenlized = list()
    with open(file, encoding='utf-8') as raw:
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenlized.append(text)
    return tokenlized


def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict

def save_dict_file(data_file, vocab_file, valid_data_file=None, test_data_file=None):
    tokens = get_tokenlized(data_file)
    if valid_data_file is None:
        valid_tokens = list()
    else:
        valid_tokens = get_tokenlized(valid_data_file)
    if test_data_file is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_data_file)
    word_set = get_word_list(tokens + valid_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if valid_data_file is None:
        sequence_len = len(max(tokens, key=len))
    else:
        sequence_len = max(len(max(tokens, key=len)), len(max(valid_tokens, key=len)))
    if test_data_file is None:
        pass
    else:
        sequence_len = max(sequence_len, len(max(test_data_file, key=len)))
    sequence_len_dict = dict()
    sequence_len_dict["sequence_len"] = str(sequence_len)

    with open(vocab_file, 'wb') as out:
        pickle.dump((word_index_dict, index_word_dict, sequence_len_dict, valid_tokens, test_tokens), out)

def get_dict_from_vocab(vocab_file, valid_data_file=None, valid_file=None, test_data_file=None, test_file=None):

    with open(vocab_file, 'rb')  as inf:
        wi_dict, iw_dict, sequence_len_dict, valid_tokens, test_tokens = pickle.load(inf)

    if valid_data_file is None:
        pass
    else:
        with open(valid_file, 'w') as outfile:
            outfile.write(text_to_code(valid_tokens, wi_dict, int(sequence_len_dict["sequence_len"])))
    if test_data_file is None:
        pass
    else:
        with open(test_file, 'w') as outfile:
            outfile.write(text_to_code(test_tokens, wi_dict, int(sequence_len_dict["sequence_len"])))

    return wi_dict, iw_dict, int(sequence_len_dict["sequence_len"]), len(wi_dict) + 1
    
def text_precess(train_text_loc, valid_file, test_file, valid_text_loc=None,test_text_loc=None):
    train_tokens = get_tokenlized(train_text_loc)
    if valid_text_loc is None:
        valid_tokens = list()
    else:
        valid_tokens = get_tokenlized(valid_text_loc)
    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc)
    word_set = get_word_list(train_tokens + valid_tokens + test_tokens)
    [word_index_dict, index_word_dict] = get_dict(word_set)

    if valid_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(valid_tokens, key=len)))
    if test_text_loc is None:
        pass
    else:
        sequence_len = max(sequence_len, len(max(test_tokens, key=len)))
        with open(test_file, 'w') as outfile:
            outfile.write(text_to_code(test_tokens, word_index_dict, sequence_len))
    if valid_text_loc is None:
        pass
    else:
        with open(valid_file, 'w') as outfile:
            outfile.write(text_to_code(valid_tokens, word_index_dict, sequence_len))

    return sequence_len, len(word_index_dict) + 1, word_index_dict, index_word_dict
