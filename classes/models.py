import tensorflow as tf
import tensorflow_addons as tfa
from transformers import TFBertModel
from tensorflow.keras.layers import Dense, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, TimeDistributed, Bidirectional, \
    BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras as k
from .utils import *
import tensorflow.keras.backend as K

class BertClassifier(tf.keras.Model):    
    def __init__(self, bert: TFBertModel, num_classes: int, bert_trainable: bool, le: MyLabelEncoder, tasks: list):
        super().__init__()
        self.bert_layer = bert
        self.num_classes = num_classes
        self.bert_layer.trainable = bert_trainable
        self.le = le
        self.tasks = tasks
        # self.dont_train = dont_train
        # self.lstm = Bidirectional(LSTM(units=768, 
        #     activation='tanh',
        #     recurrent_activation='sigmoid',
        #     recurrent_dropout=0.0,
        #     unroll=False,
        #     use_bias=True,
        #     return_sequences=True, 
        #     dropout=0.5, 
        #     # recurrent_dropout=0.5, 
        #     kernel_initializer=tf.keras.initializers.he_normal()))   
        self.dropout = Dropout(0.2)
        self.batch_norm = BatchNormalization(axis=-1)
        self.classifier = OrderedDict([(key, TimeDistributed(Dense(len(le[key]), activation='relu'))) for key in le])
        self.crf = OrderedDict([(key, tfa.layers.CRF(len(le[key]), use_kernel=False)) for key in le])
        
        # norm classifier
        if 'norm' in le:
            self.norm_conv_1d = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')
            self.norm_maxpooling_1d = MaxPooling1D(pool_size=10)
            self.norm_lstm = LSTM(units=768, 
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 recurrent_dropout=0.0,
                 unroll=False,
                 use_bias=True,
                 return_sequences=True, 
                 dropout=0.5, 
                 # recurrent_dropout=0.5, 
                 kernel_initializer=tf.keras.initializers.he_normal())
            self.norm_classifier = Dense(units=len(le['norm']), activation='softmax')
            self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        
    def loss(self, y_true, y_pred, dash_symbol, pad_symbol):
        if self.le.use_dash:
            ignore_symbols = tf.constant([dash_symbol])
            mask = tf.cast(tf.reduce_prod(tf.map_fn(lambda y: tf.cast(tf.not_equal(y_true, y), tf.int32), elems=ignore_symbols), axis=0), tf.bool)
            y_t = tf.ragged.boolean_mask(y_true, mask).to_tensor()
            y_p = tf.ragged.boolean_mask(y_pred[:,:,:-1], mask).to_tensor() 
            
            is_empty = tf.logical_or(tf.equal(tf.size(y_t), 0), tf.equal(tf.size(y_p), 0))
            
            if is_empty:
                log_likelihood, self.transitions = tfa.text.crf.crf_log_likelihood(
                    y_pred,
                    y_true,
                    self.sequence_lengths,
                    transition_params=self.transitions,
                )
            else:
                sequence_lengths = tf.math.reduce_sum(tf.cast(tf.math.not_equal(y_t, pad_symbol), dtype=tf.int32), axis=-1)
                log_likelihood, self.transitions = tfa.text.crf.crf_log_likelihood(
                    y_p,
                    y_t,
                    sequence_lengths,
                    transition_params=self.transitions[:-1,:-1],
                )
            # ignore_symbols = tf.constant([dash_symbol])
            # mask = tf.cast(tf.reduce_prod(tf.map_fn(lambda y: tf.cast(tf.equal(y_true, y), tf.int32), elems=ignore_symbols), axis=0), tf.bool)
            # y_p = y_pred * tf.cast(tf.expand_dims(~mask, -1), tf.float32)
            # y_p = y_p + tf.one_hot(tf.cast(mask, tf.int32) * dash_symbol, dash_symbol + 1) * tf.cast(tf.expand_dims(mask, -1), tf.float32)
            # log_likelihood, self.transitions = tfa.text.crf.crf_log_likelihood(
            #     y_p,
            #     y_true,
            #     self.sequence_lengths,
            #     transition_params=self.transitions,
            # )
        else:
            log_likelihood, self.transitions = tfa.text.crf.crf_log_likelihood(
                y_pred,
                y_true,
                self.sequence_lengths,
                transition_params=self.transitions,
            )
        return tf.reduce_mean(-log_likelihood)

    def my_f1_score(self, y_true, y_pred, lbls, avg):
        f1 = f1_score(y_true, y_pred, labels=lbls, average=avg, zero_division=0)
        return f1
    
    def f1(self, y_true, y_pred, num_classes, ignore_symbols=None):
        mask1 = tf.cast(tf.reduce_prod(tf.map_fn(lambda y: tf.cast(tf.not_equal(y_true, tf.convert_to_tensor(y)), tf.int32), elems=tf.convert_to_tensor(ignore_symbols)), axis=0), tf.bool)
        tp = tf.reduce_sum(tf.cast(tf.boolean_mask(tf.equal(y_true, y_pred), mask1), tf.float32))
        tp_fn = tf.reduce_sum(tf.cast(mask1, tf.float32))
        mask2 = tf.reduce_prod(tf.map_fn(lambda y: tf.cast(tf.not_equal(y_pred, tf.convert_to_tensor(y)), tf.int32), elems=tf.convert_to_tensor(ignore_symbols)), axis=0)
        tp_fp = tf.cast(tf.reduce_sum(mask2), tf.float32)                          
        p = tp / (tp_fp + K.epsilon())
        r = tp / (tp_fn  + K.epsilon())
        f1 = 2 * (p * r) / (p + r + K.epsilon())
        
        # lbls = [x for x in range(num_classes)]
        # mask = tf.cast(tf.reduce_prod(tf.map_fn(lambda y: tf.cast(tf.not_equal(lbls, tf.convert_to_tensor(y)), tf.int32), elems=ignore_symbols), axis=0), tf.bool)
        # lbls = tf.convert_to_tensor(tf.boolean_mask(lbls, mask))
        # f1 = tf.py_function(func=self.my_f1_score, inp=[tf.reshape(y_true, shape=(-1,)), tf.reshape(y_pred, shape=(-1,)), lbls, 'micro'], Tout=tf.float32)
        return f1
        
    def call(self, input_ids, attention_mask, labels=None, training=None, dont_train=None):
        bert_output = self.bert_layer(input_ids, attention_mask=attention_mask)
        
        bert_output = bert_output['hidden_states'][-1]
        # bert_output = tf.reduce_sum(bert_output['hidden_states'][-4:], axis=0)
        # bert_output = tf.concat(bert_output['hidden_states'][-2:], axis=-1)
        
        # bert_output = self.dropout(bert_output, training)
        # bert_output = self.batch_norm(bert_output, training)
        
        # bert_output = self.lstm(bert_output)
        cls_output = OrderedDict()
        out_dict = {i: key for i, key in enumerate(list(self.le))}
        
        outputs = OrderedDict()
        potentials = OrderedDict()
        if not labels is None:
            loss = OrderedDict()
            f1s = OrderedDict()
            # tf.print(dont_train)
            for i, key in enumerate(self.le):
                if not dont_train is None and key in dont_train:
                    continue
                y_true = labels[:,:,i]
                num_classes = len(self.le[key])
                if i > 10:
                    #  if training else tf.argmax(outputs[tasks[0]], axis=-1, output_type=tf.int32)
                    mask = tf.cast(tf.reduce_prod(tf.map_fn(lambda y: tf.cast(tf.not_equal(y_true, tf.convert_to_tensor(y)), tf.int32), 
                                                            elems=tf.constant([self.le.get_code(key, '<pad>'), 
                                                                               self.le.get_code(key, 'O')] + ([self.le.get_code(key, '-')] if self.le.use_dash else []),
                                                                               )), axis=0), tf.bool)
                                                                               
                    # y_true = y_true * tf.cast(mask, tf.int32) + ((y_true - y_true) + self.le.get_code(key, '-')) * tf.cast(~mask, tf.int32)

                    bert_output = bert_output * tf.cast(tf.expand_dims(mask, -1), tf.float32)
                    # bert_output = tf.ragged.boolean_mask(bert_output, mask).to_tensor()
                    # y_t = tf.ragged.boolean_mask(y_true, mask).to_tensor()
                    
                    # cls_output[key] = self.classifier[key](bert_output)
                    # self.decoded_sequence, output, self.sequence_lengths, self.transitions = self.crf[key](cls_output[key])

                    # f1s[key] = self.f1(y_t, self.decoded_sequence, len(self.le[key]), [le.get_code(key, '<pad>'), 
                    #                                                                  le.get_code(key, 'O'),
                    #                                                                  le.get_code(key, '-')])
                    # loss[key] = self.loss(y_t, output, le.get_code(key, '-'), le.get_code(key, '<pad>'))
                    # outputs[key] = tf.one_hot(self.decoded_sequence, num_classes)
                    # potentials[key] = output
                    
                    # bert_output = tf.concat([bert_output, tf.one_hot(y_true if training else self.decoded_sequence, num_classes)], -1)
                    # bert_output = tf.one_hot(y_true, num_classes)
                
                # print(key, bert_output)
                cls_output[key] = self.classifier[key](bert_output)
                self.decoded_sequence, output, self.sequence_lengths, self.transitions = self.crf[key](cls_output[key])

                f1s[key] = self.f1(y_true, self.decoded_sequence, len(self.le[key]), [self.le.get_code(key, '<pad>'), 
                                                                                      self.le.get_code(key, 'O')] + ([self.le.get_code(key, '-')] if self.le.use_dash else []))

                loss[key] = self.loss(y_true, output, self.le.get_code(key, '-') if self.le.use_dash else None, self.le.get_code(key, '<pad>'))
                # if not dont_train is None and key in dont_train:
                #     loss[key] = tf.constant([0.0])
                # else:
                #     loss[key] = self.loss(y_true, output, self.le.get_code(key, '-') if self.le.use_dash else None, self.le.get_code(key, '<pad>'))
                # self.f1_score[key] = self.f1(y_true, output, num_classes)
                outputs[key] = tf.one_hot(self.decoded_sequence, num_classes)
                potentials[key] = output
            # print(list(outputs))
            
            return outputs, loss, f1s
        else:    
            transitions = OrderedDict()
            for i, key in enumerate(self.le):
                if not dont_train is None and key in dont_train:
                    continue
                num_classes = len(self.le[key])
                if i > 10:
                    mask = tf.cast(tf.reduce_prod(tf.map_fn(lambda y: tf.cast(tf.not_equal(tf.argmax(outputs[self.tasks[0]], axis=-1, output_type=tf.int32), tf.convert_to_tensor(y)), tf.int32), 
                                                            elems=tf.constant([self.le.get_code(key, '<pad>'), 
                                                                               self.le.get_code(key, 'O')] + ([self.le.get_code(key, '-')] if self.le.use_dash else []),
                                                                               )), axis=0), tf.bool)
                    bert_output = bert_output * tf.cast(tf.expand_dims(mask, -1), tf.float32)
                    # bert_output = bert_output * mask
                    # bert_output = tf.concat([bert_output, tf.one_hot(self.decoded_sequence, num_classes)], -1)
                    # bert_output = tf.one_hot(y_true, num_classes)
                
                # print(key, bert_output)
                cls_output[key] = self.classifier[key](bert_output)
                num_classes = len(self.le[key])
                # y_true = labels[:,:,i,:num_classes]
                # self.f1_score[key] = self.f1(y_true, output, num_classes)
                # f1s[key] = self.f1(y_true, output, self.transitions, len(self.le[key]))
                self.decoded_sequence, output, self.sequence_lengths, self.transitions = self.crf[key](cls_output[key])
                potentials[key] = output
                # if i > 0:
                #     self.decoded_sequence = self.decoded_sequence * tf.cast(mask, tf.int32) + ((self.decoded_sequence - self.decoded_sequence) + self.le.get_code(key, 'O')) * tf.cast(~mask, tf.int32)
                outputs[key] = tf.one_hot(self.decoded_sequence, num_classes)
                transitions[key] = self.transitions
            return outputs, transitions, None
            

@tf.function(experimental_relax_shapes=True)
def train_one_step(model, optimizer, input_ids, attention_mask, y, dont_train=None, experimental_relax_shapes=True):
    with tf.GradientTape() as tape:
        if dont_train is None:
            logits, loss, f1s = model(input_ids, attention_mask, y)
        else:
            logits, loss, f1s = model(input_ids, attention_mask, y, dont_train=dont_train)
    gradients = tape.gradient([loss[key] for key in loss], model.trainable_variables)
    # print(type(gradients), len(gradients))
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, logits, f1s
    

@tf.function(experimental_relax_shapes=True)
def predict_one_step(model, val_input_ids, val_attention_mask, val_y, dont_train=None, experimental_relax_shapes=True):
    logits, loss, f1s = model(val_input_ids, val_attention_mask, val_y, training=False, dont_train=dont_train)
    # tf.print(f1s)
    return loss, logits, f1s

def get_batch_entities(keys, ys, label, use_cls, max_length, le, dev_meta_data, to_be_saved=None, return_df=False):
    res = []
    columns = ['clinical_case', 'mark', 'label', 'offset', 'span', 'start_pos_pred', 'end_pos_pred']
    for key, output in zip(keys, ys):
        id = key[0]
        orig_sent = dev_meta_data[key]
        output = output[1:-1] if use_cls else output
        offset = 0
        last_lbl = 'none'
        for j in range(min(len(orig_sent), max_length)):
            start, end, count, token = orig_sent[j][:4]
            # out = sorted([le.inv_dictionary[label][x] for x in output[offset:offset+count]])
            out = [le.inv_dictionary[label][x] for x in output[offset:offset+count]]
            # print(out)
            # Just in case the sentence has been truncated
            if len(out) > 0:
                lbl = out[0]
                if lbl.startswith('B-') or lbl.startswith('U-'):
                    if not to_be_saved is None:
                        to_be_saved[label][id].append(['T%s', lbl[2:], start, end, token])
                    elif return_df:
                        res.append([id + '.ann', 'T%s' % 0,  lbl[2:],  '%s %s' % (start, end), token, start, end])
                    else:
                        res.append([id, lbl[2:], start, end])
                elif lbl.startswith('I-') or lbl.startswith('L-'):
                    if last_lbl[:2] in ['B-', 'I-']:
                        if not to_be_saved is None:
                            to_be_saved[label][id][-1][-2] = end
                            to_be_saved[label][id][-1][-1] += ' ' + token
                        elif return_df:
                            res[-1][-1] = end
                            res[-1][-3] += ' ' + token
                            res[-1][3] = '%s %s' % (res[-1][5], end)
                        else:
                            res[-1][-1] = end
                    else:
                        if not to_be_saved is None:
                            to_be_saved[label][id].append(['T%s', lbl[2:], start, end, token])
                        elif return_df:
                            res.append([id + '.ann', 'T%s' % 0,  lbl[2:],  '%s %s' % (start, end), token, start, end])
                        else:
                            res.append([id, lbl[2:], start, end])
                offset += count
                last_lbl = lbl
    return to_be_saved if not to_be_saved is None else [tuple(x) for x in res] if not return_df else pd.DataFrame(res, columns=columns)

# def get_entities(keys, y_pred, y_true, gen_output=False):
#     true_entities = defaultdict(list)
#     pred_entities = defaultdict(list)
#     to_be_saved = {}
#     for label in list(le):
#         y_pred_res = get_batch_entities(keys, y_pred, label, gen_output=gen_output)
#         if not gen_output:
#             y_true_res = get_batch_entities(keys, y_true, label, gen_output=gen_output)
#             for key in y_pred_res:
#                 if not gen_output:
#                     true_entities[key].extend(y_true_res[key])
#                 pred_entities[key].extend(y_pred_res[key])
#         else:
#             to_be_saved.update(y_pred_res)
#     return to_be_saved if gen_output else (true_entities, pred_entities)
