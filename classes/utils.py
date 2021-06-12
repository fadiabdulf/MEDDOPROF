import string

class Token:
    def __init__(self, text, i, idx):
        self._text = text
        self.i = i
        self.idx = idx
    
    def __str__(self):
        return self._text
    
    def __repr__(self):
        return self._text
    
    def __len__(self):
        return len(self._text)
    
    def text(self):
        return self._text

class Doc:
    def __init__(self, text, tokenizer=None, encod=None):
        self.text = text
        self.encod = encod
        if encod is None:
            self.encod = tokenizer(text).encodings[0]
        self.li = [Token(text[idx:end], i, idx) for i, (idx, end) in enumerate(self.encod.offsets)]
    
    def __str__(self):
        return self._text
    
    def __repr__(self):
        return self._text
    
    def __len__(self):
        return len(self.li)

    def __getattr__(self, method):
        return getattr(self.li, method)
    
    def __getitem__(self, item):
        return self.li[item]

    
    
def my_tokenizer(text, tokenizer=None, encod=None):
    return Doc(text, tokenizer=tokenizer, encod=encod)

import spacy
from spacy.training import offsets_to_biluo_tags
from collections import defaultdict
# nlp = {'en': spacy.load("en_core_web_sm"), 'es': spacy.load("es_core_news_md")}
# nlp = {'en': spacy.load("en_core_web_trf"), 'es': spacy.load('es_dep_news_trf')}
# nlp = defaultdict(lambda: spacy.load('xx_ent_wiki_sm'))
nlp = defaultdict(lambda: spacy.load('xx_sent_ud_sm'))
def label(row, columns, labeling_scheme='BIO', use_dash_tag=True, spacy_lang='es'):
    """
    """
    doc = nlp[spacy_lang](row['text'])
    for col in columns:
        ents = sorted(row[col], key=lambda s_e_t: s_e_t[2])
        if False:
            if row['id'] == 'casos_clinicos_infecciosas78':
                ents = sorted(row[col][1:], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'casos_clinicos_profesiones39':
                ents = sorted([(s, e, t) for s, e, t in row[col] if s != 2591], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'casos_clinicos_profesiones99':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 4917 and t == 'FAMILIAR')], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_atencion_primaria160':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 2290 and e == 2317)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_dermatologia486':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 525 and (e == 553 or e == 571))], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_psiquiatria169':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 1331 and e == 1367)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_psiquiatria17':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 2141 and e == 2169)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_psiquiatria224':
                ents = sorted([(s, e if not e == 139 else 152, t) for s, e, t in row[col] if not(s == 132 and e == 152)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_psiquiatria237':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 258 and e == 284)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_psiquiatria287':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 6761 and e == 6789)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_psiquiatria297':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 441 and e == 462)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_psiquiatria481':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 780 and e == 823)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_psiquiatria59':
                ents = sorted([(s, e if not e == 481 else 500, t) for s, e, t in row[col] if not(s == 474 and e == 500)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'caso_clinico_psiquiatria8':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 561 and e == 607)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'cc_onco222':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 356 and e == 379)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'cc_onco229':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 28 and e == 57)], key=lambda s_e_t: s_e_t[2])
            elif row['id'] == 'cc_onco687':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 196 and e == 225)], key=lambda s_e_t: s_e_t[2])
        else:
            id = row['id'].split('/')[-1]
            if id == 'casos_clinicos_infecciosas78':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 468 and e == 498)], key=lambda s_e_t: s_e_t[2])
            elif id == 'casos_clinicos_profesiones39':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 4917 and e == 4934 or s == 2537 and e == 2605)], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_atencion_primaria160':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 2290 and e == 2317)], key=lambda s_e_t: s_e_t[2])
            elif id == 'casos_clinicos_profesiones99':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 4917 and e == 4934)], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_dermatologia486':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 525 and (e == 553 or e == 571))], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_psiquiatria169':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 1331  and e == 1367)], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_psiquiatria17':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 2141  and e == 2169)], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_psiquiatria224':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 118  and e == 139)], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_psiquiatria237':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 258  and e == 284)], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_psiquiatria287':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 6761 and e == 6789)], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_psiquiatria297':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 441 and e == 462)], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_psiquiatria481':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 780 and e == 823)], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_psiquiatria59':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 464 and e == 481)], key=lambda s_e_t: s_e_t[2])
            elif id == 'caso_clinico_psiquiatria8':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 561 and e == 579)], key=lambda s_e_t: s_e_t[2])
            elif id == 'cc_onco222':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 356 and e == 379)], key=lambda s_e_t: s_e_t[2])
            elif id == 'cc_onco229':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 28 and e == 57)], key=lambda s_e_t: s_e_t[2])
            elif id == 'cc_onco687':
                ents = sorted([(s, e, t) for s, e, t in row[col] if not(s == 196 and e == 225)], key=lambda s_e_t: s_e_t[2]) 
        i, l, j = 0, 0, 0
        try:
            lbls = offsets_to_biluo_tags(doc, ents)
            # fix '-'
            j = 0
            last = False
            for i, l in enumerate(lbls):
                if l[:2] in ['B-', 'U-'] and j < len(ents) - 1:
                    j += 1
                elif l == '-':
                    if i - 1 > 0 and lbls[i - 1] == 'O' or i == 0:
                        if i + 1 < len(lbls) and lbls[i + 1] != 'O' or i + 1 >= len(lbls):
                            lbls[i] = 'B-' + ents[j][-1]
                        else:
                            lbls[i] = 'U-' + ents[j][-1]
                    elif i - 1 > 0 and lbls[i - 1][:2] in ['B-', 'I-']:
                        if i + 1 < len(lbls) and lbls[i + 1] == 'O' or i + 1 >= len(lbls):
                            lbls[i] = 'L-' + ents[j][-1]
                        else:
                            lbls[i] = 'I-' + ents[j][-1]
                    if j < len(ents) - 1:
                        j += 1
                        last = True
                if j >= len(ents):
                    break
            # lbls = offsets_to_biluo_tags(nlp[spacy_lang].make_doc(row['text']), ents)
            sents = {}
            ents = {}
            for sent in doc.sents:
                if sent.text.strip():
                    sents[sent.start_char] = [(token.idx, token.text) for token in sent if not token.is_space]
                    if not use_dash_tag:
                        ents[sent.start_char] = [lbl if lbl != '-' else 'O' for tok, lbl in zip(sent, lbls[sent[0].i:sent[0].i+len(sent)]) if not tok.is_space]
                    if labeling_scheme.lower() in ['bio', 'iob']:
                        ents[sent.start_char] = [lbl.replace('L-', 'I-').replace('U-', 'B-') for tok, lbl in zip(sent, lbls[sent[0].i:sent[0].i+len(sent)]) if not tok.is_space]
                    else:
                        ents[sent.start_char] = [lbl for tok, lbl in zip(sent, lbls[sent[0].i:sent[0].i+len(sent)]) if not tok.is_space]

            if not 'sents' in row:
                row['sents'] = sents
            row['%s_lbls' % col] = ents
        except Exception as ex:
            print(ex)
            print(i, l)
            print(j, ents)
            print(col, row['id'], sorted(row[col], key=lambda s_e_t: s_e_t[0]))
    return row


import pandas as pd
# split long sentences to n parts, every part less than max_length
def split_long_sentence(df, max_length, remove=False):
    total = len(df[['id', 'key']].apply(lambda y: ' '.join([str(x) for x in y]), axis=1).value_counts())
    df.index  = df[['id', 'key', 'start']]
    for key, group in tqdm(df.groupby(['id', 'key'], sort=False), total=total):
        if group['count'].sum() > max_length:
            count = 0
            last_pos = 0
            new_start = None
            for i, row in group.iterrows():
                count += row['count']
                new_pos = count % max_length
                if new_pos != last_pos:
                    last_pos = new_pos
                    if remove:
                        new_start = -1
                    else:
                        new_start = row['start']
                if not new_start is None:
                    df.at[(row['id'], row['key'], row['start']), 'key'] = new_start
    return df[df['key'] != -1]


import pandas as pd
def convert_to_token_level(data, columns=None, tokenizer=None, spacy_lang='es', lower_case=False):
    token_level_data = []
    if columns is None:
        out_cols = ['id', 'key', 'token', 'sub_tokens', 'start', 'end', 'count']
        cols = [col for col in data.columns if col.endswith('lbls')]
        for i, row in tqdm(data.iterrows(), total=len(data)):
            id = row['id']
            sents = row['sents']
            lbls = [row[col] for col in cols]
            for offset in sents:
                ls = [lbls[j][offset] for j in range(len(cols))]
                for j, (start, token) in enumerate(sents[offset]):
                    token = token.lower() if lower_case else token
                    if tokenizer is None:
                        sub_tokens = []
                    else:    
                        sub_tokens = tokenizer(token, add_special_tokens=False)['input_ids']
                    token_level_data.append([id, offset, token, sub_tokens, start, start + len(token), len(sub_tokens)] + [s_l_lbls[j] for s_l_lbls in ls])
        token_level_data = pd.DataFrame(token_level_data, columns=out_cols + cols)
    else:
        cols = [col for col in data.columns if col.endswith('lbls')]
        for i, row in tqdm(data.iterrows(), total=len(data)):
            id = row['id']
            sents = row['sents']
            lbls = [row[col] for col in cols]
            for offset in sents:
                ls = [lbls[j][offset] for j in range(len(cols))]
                doc = nlp[spacy_lang](' '.join([x for _, x in sents[offset]]))
                k = 0
                for j, (start, token) in enumerate(sents[offset]):
                    while doc[k].is_space:
                        k += 1
                    fs = get_fs(doc[k])
                    token_level_data.append([id, offset, start, start + len(token), fs, token, doc[k].tag_] + [s_l_lbls[j] for s_l_lbls in ls])
        token_level_data = pd.DataFrame(token_level_data, columns=columns + cols)
    return token_level_data

def get_fs(token):
    return  {'is_punct': token.is_punct, 'shape_': token.shape_, 'pos_': token.pos_, 'lemma_': token.lemma_, 
             'ent_type_': token.ent_type_, 'ent_iob_': token.ent_iob, 'like_email': token.like_email, 
             'dep_': token.dep_, 'is_stop': token.is_stop, 'is_alpha': token.is_alpha}
    
# class MyLabelEncoder:
#     def __init__(self, dictionary):
#         if isinstance(dictionary, dict):
#             self.dictionary = dictionary
#         elif isinstance(dictionary, list):
#             self.dictionary = {dictionary[i]: i for i in range(len(dictionary))}
#         self.inv_dictionary = {self.dictionary[key]: key for key in self.dictionary}
#         self.classes_ = list(dictionary)
        
#     def transform(self, X):
#         return [self.dictionary[x] for x in X]
    
#     def inverse_transform(self, X):
#         return [self.inv_dictionary[x] for x in X]

class MyLabelEncoder:
    def __init__(self, dictionary, labeling_scheme, use_dash, multi=True):
        self.multi = multi
        self.use_dash = use_dash
        if not multi:
            self.names = None
            if isinstance(dictionary, dict):
                self.dictionary = dictionary
            elif isinstance(dictionary, list):
                if labeling_scheme.lower() in ['bio', 'iob']:
                    dictionary = ['O', '<pad>'] + [x + y for y in dictionary for x in ['B-', 'I-']] + (['-'] if use_dash else [])
                else:
                    dictionary = ['O', '<pad>'] + [x + y for y in dictionary for x in ['B-', 'I-', 'U-', 'L-']] + (['-'] if use_dash else [])
                self.dictionary = {dictionary[i]: i for i in range(len(dictionary))}
            self.inv_dictionary = {self.dictionary[key]: key for key in self.dictionary}
            self.classes_ = list(dictionary)
        else:
            self.names = list(dictionary)
            self.dictionary = OrderedDict()
            self.inv_dictionary = OrderedDict()
            for key in dictionary:
                if isinstance(dictionary[key], dict):
                    self.dictionary[key] = dictionary[key]
                elif isinstance(dictionary[key], list):
                    if labeling_scheme.lower() in ['bio', 'iob']:
                        self.dictionary[key] = ['O', '<pad>'] + [x + y for y in dictionary[key] for x in ['B-', 'I-']] + (['-'] if use_dash else [])
                    else:
                        self.dictionary[key] = ['O', '<pad>'] + [x + y for y in dictionary[key] for x in ['B-', 'I-', 'U-', 'L-']] + (['-'] if use_dash else [])
                    self.dictionary[key] = {self.dictionary[key][i]: i for i in range(len(self.dictionary[key]))}
                self.inv_dictionary[key] = {self.dictionary[key][k]: k for k in self.dictionary[key]}
            self.names = list(self.dictionary)
        
        
    def transform(self, X):
        if self.multi:
            if isinstance(X[0], np.ndarray) or isinstance(X[0], tuple) or isinstance(X[0], list):
                return [[self.dictionary[self.names[i]][y] for i, y in enumerate(x)] for x in X]
            else:
                return [self.dictionary[x] for x in X]
        else:
            return [self.dictionary[x] for x in X]
    
    def inverse_transform(self, X):
        if self.multi:
            if isinstance(X[0], np.ndarray) or isinstance(X[0], tuple) or isinstance(X[0], list):
                return [self.inv_dictionary[i][x] for x in X for i, y in enumerate(x)]
        return [self.inv_dictionary[x] for x in X]

    def get_code(self, key, symbol, norm=False):
        if norm:
            return self.dictionary[key][symbol][2:]
        return self.dictionary[key][symbol]

    def get_symbol(self, key, code, norm=False):
        if norm:
            return (self.inv_dictionary[key][code] - 2) / 2
        return self.inv_dictionary[key][code]
    
    def __getitem__(self, key):
        return self.dictionary[key]
    
    def values(self):
        return self.dictionary.values()

    def items(self):
        return self.dictionary.items()

    def __contains__(self, item):
        return item in self.dictionary

    def __iter__(self):
        return iter(self.dictionary)
    
    def __repr__(self):
        return str(self.dictionary)
    
    def __len__(self):
        if self.multi:
            return max([len(self.dictionary[k]) for k in self.dictionary])
        else:
            return len(self.dictionary)

# test case
# len(my_tokenizer("Fadi, Hello?"))

import random
import numpy as np
from collections import OrderedDict, defaultdict
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def _batching(batch, extend, max_len, pad_token, batch_ordered_keys, batch_ordered_ids, batch_ordered_masks, batch_ordered_labels):
    keys1, new_sents = zip(*[(sent[0], ([s[0] for s in sent[1]], [1] * len(sent[1]))) for sent in batch])
    ids, msks = zip(*new_sents)
    keys2, new_lbls = zip(*[(sent[0], [s[1] for s in sent[1]]) for sent in batch])
    
    if extend:
        batch_ordered_keys.extend(keys1)
        batch_ordered_ids.extend(ids)
        batch_ordered_masks.extend(msks)
        batch_ordered_labels.extend(new_lbls)
    else:
        maxlen = min(max_len, max([len(s) for s in ids]))
        ids = [list(x) for x in pad_sequences(ids, maxlen=maxlen, padding='post', truncating='post', value=pad_token)]
        msks = [list(x) for x in pad_sequences(msks, maxlen=maxlen, padding='post', truncating='post', value=0)]
        new_sents = zip(ids, msks)
        new_lbls = [list(y) for y in pad_sequences(new_lbls, maxlen=maxlen, padding='post', truncating='post', value='<pad>', dtype=object)]
        batch_ordered_keys.append(keys1)
        batch_ordered_ids.append(ids)
        batch_ordered_masks.append(msks)
        batch_ordered_labels.append(new_lbls)

    return batch_ordered_keys, batch_ordered_ids, batch_ordered_masks, batch_ordered_labels

def batching(samples, batch_size, extend=False, pad_token=0, max_len=128, mix=True):
    
    # List of batches that we'll construct.
    batch_ordered_keys = []
    batch_ordered_ids = []
    batch_ordered_masks = []
    batch_ordered_labels = []

    print('Creating batches of size {:}'.format(batch_size))

    # Loop over all of the input samples...   
    if not mix:
        for select in trange(0, len(samples), batch_size):
            to_take = batch_size
            batch = samples[select:(select + to_take)]
            
            batch_ordered_keys, batch_ordered_ids, batch_ordered_masks, batch_ordered_labels = \
                _batching(batch, extend, max_len, pad_token, batch_ordered_keys, batch_ordered_ids, batch_ordered_masks, batch_ordered_labels)
    else:
        while len(samples) > 0:
            
            # Report progress.
            if ((len(batch_ordered_ids) % 1000) == 0):
                print('  Selected {:,} batches.'.format(len(batch_ordered_ids)))

            # `to_take` is our actual batch size. It will be `batch_size` until 
            # we get to the last batch, which may be smaller. 
            to_take = min(batch_size, len(samples))

            # Pick a random index in the list of remaining samples to start
            # our batch at.
            select = random.randint(0, len(samples) - to_take)

            # Select a contiguous batch of samples starting at `select`.
            batch = samples[select:(select + to_take)]

            # Each sample is a tuple--split them apart to create a separate list of 
            # sequences and a list of labels for this batch.
            
            batch_ordered_keys, batch_ordered_ids, batch_ordered_masks, batch_ordered_labels = \
                _batching(batch, extend, max_len, pad_token, batch_ordered_keys, batch_ordered_ids, batch_ordered_masks, batch_ordered_labels)

            # Remove these samples from the list.
            del samples[select:select + to_take]

        # print('\n  DONE - {:,} batches.'.format(len(batch_ordered_sentences)))
    return batch_ordered_keys, batch_ordered_ids, batch_ordered_masks, batch_ordered_labels


from tensorflow import keras 
from sklearn_crfsuite import metrics
from tqdm.auto import tqdm, trange


class Metrics(keras.callbacks.Callback):
    def __init__(self, data, data_size, le, model, filepath):
        self.data = data
        self.data_size = data_size
        self.le = le
        # self.labels = [t for t in self.le.classes_ if t != 'O' and t != 'PAD']
        self.model = model
        self.filepath = filepath

    def on_train_begin(self, data):
        self._data = data
        self.best_score = 0.0

    def on_epoch_end(self, batch, logs={}):
        y_pred = []
        y_true = []
        for x, y in tqdm(self.data, total=self.data_size):
            res = self.model(x).numpy()
            y_pred.extend([self.le.inverse_transform(p) for p in res.argmax(-1)])
            y_true.extend([self.le.inverse_transform(y1) for y1 in y.numpy().argmax(-1)])
        # score = metrics.flat_f1_score(y_true, y_pred, average='weighted', labels=self.labels)
        # score = metrics.flat_f1_score(y_true, y_pred, average='macro', labels=self.labels)
        print()
        print("model score: %s" % score)
        # y_pred = set_output(dev_input, y_pred)
        # dev_output = Utilis.df_to_files(dev_input, y_pred)
        # scores = Utilis.my_score(dev_input, dev_output, details=False)
        # score = scores.iloc[-1]['F1']
        if score > self.best_score:
            print("best score: %s" % score)
            # print(dict(scores.iloc[-1]))
            self.model.save_weights(self.filepath.replace('{val_acc:.5f}', 'model_weights_%0.5f') % score)
            self.best_score = score
        print('\n')
        return 

    def get_data(self):
        return self._data