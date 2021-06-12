import spacy
from spacy.lang.en import English
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import re
from abc import ABC, abstractmethod

class BaseTokenizer(object):
    """
    """
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def tokenize(self):
        """
        """

class Tokenizer(BaseTokenizer):
    """
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md", disable=["ner"])
        # # Construction 2
        # nlp = English()
        # # Create a Tokenizer with the default settings for English
        # # including punctuation rules and exceptions
        # self.nlp = nlp.Defaults.create_tokenizer(nlp)
        self.sentenc_sp = re.compile('\s*\n{3,}\s*|\s*_{2,}\s*', re.M)
        self.line_sp = re.compile('\s{2,}', re.M)
        self.re_rep = r'\[\*\*(.+?)\*\*\]'
        self.re_sub = r'\1'
    
    def pre_process(self, text, entities):
        ents = [(start - i * 6, end - (i * 6 + 6), ent_type) for i, (start, end, ent_type) in enumerate(entities)]
        return re.sub(self.re_rep, self.re_sub, text), ents
        # text = row['text']
        # res = []
        # offset = 0
        # for start, end, ent_type in row['entities']:
        #     res.append(text[offset:start])
        #     res.append(text[start:end][3:-3])
        #     offset = end
        # res.append(text[offset:])
        # return ''.join(res)

    def my_split(self, text, re_sep, entities=None):
        offset = 0
        if entities is None:
            res = []
            sps = []
            for x in re.finditer(re_sep, text):
                res.append(text[offset:x.span()[0]])
                sps.append(x.group())
                offset = x.span()[1]
            if len(text[offset:]) > 0:
                res.append(text[offset:])
            sps.append('')
            return res, sps, None
        else:
            res = []
            sps = []
            ents = []
            k = 0
            for x in re.finditer(re_sep, text):
                res.append(text[offset:x.span()[0]])
                sps.append(x.group())
                old_offset = offset
                offset = x.span()[1]
                ents.append([])
                while k < len(entities) and entities[k][0] < offset:
                    y = entities[k]
                    k += 1
                    ents[-1].append([y[0] - old_offset, y[1] - old_offset, y[2]])
            if len(text[offset:]) > 0:
                res.append(text[offset:])
            ents.append([])
            while k < len(entities):
                y = entities[k]
                k += 1
                ents[-1].append([y[0] - offset, y[1] - offset, y[2]])
            sps.append('')
            return res, sps, ents

    def chunking(self, text, doc, entities):
        # j = 0
        # k = 0
        # l = 0
        # new_entities = [[-1, -1, None]] if len(entities) > 0 else []
        new_chunks = []
        chunks = list(doc.noun_chunks)
        for ch in chunks:
            ch_start = ch.start
            l = ch_start
            # print("          len: {}, ch: {}, ch2: {}".format(ch.end-ch.start, ch, list(doc[ch.start:ch.end])))
            for i, t in enumerate(doc[ch.start:ch.end]):
                # print(t, new_chunks)
                l += 1
                if t.is_space:
                    if len(doc[ch_start:ch_start + i].text.strip()) != 0:
                        new_chunks.append((doc[ch_start].idx, t.idx - 1))
                    ch_start = l
            if len(doc[ch_start:ch.end].text.strip()) != 0:
                new_chunks.append((doc[ch_start].idx, doc[ch.end-1].idx + len(doc[ch.end-1])))
            # print(new_chunks)
        new_entities = []
        for start, end, ent_type in entities:
            sub_text = text[start:end]
            strip_sub_text = sub_text.strip()
            idx = sub_text.index(strip_sub_text)
            new_entities.append((start + idx, start + idx + len(strip_sub_text), ent_type))
        # tokens = []
        # for i, t in enumerate(doc):
        #     start = t.idx
        #     end = t.idx + len(t.text)
        #     if j < len(entities):
        #         if start >= entities[j][0] and end <= entities[j][1]:
        #             if new_entities[-1][0] == -1:
        #                 new_entities[-1][0] = l
        #                 new_entities[-1][1] = l
        #                 new_entities[-1][2] = entities[j][2] 
        #             if not t.is_space:
        #                 new_entities[-1][1] += 1
        #         if end >= entities[j][1] and new_entities[-1][0] != -1 and j < len(entities):
        #             if j < len(entities) - 1:
        #                 new_entities.append([-1, -1, None])
        #             j += 1 
        #     if k < len(chunks):
        #         if i == chunks[k].start:
        #             new_chunks.append([t.is_space, l, l + 1])
        #         elif i > chunks[k].start and i < chunks[k].end:
        #             if t.is_space != new_chunks[-1][0]:
        #                 new_chunks.append([t.is_space, l, l + 1])
        #             else:
        #                 new_chunks[-1][2] += 1
        #         if i == chunks[k].end and k < len(chunks):
        #             k += 1
        #             new_chunks.append([True, l, l + 1])
        #     if not t.is_space:
        #         l += 1
        #         tokens.append(t.text)
        # return [[start, end] for is_space, start, end in new_chunks if not is_space], entities, tokens
        return new_chunks, new_entities, [(t.idx, t.text) for t in doc if not t.is_space]

    def tokenize(self, df):
        res = []
        # cols = ['sentenc_id', 'text', 'tokens', 'chunks']
        cols = ['sentenc_id', 'text', 'tokens', 'chunks', 'entities']
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing: "):
            text, entities = self.pre_process(row['text'], row['entities'])
            # doc = self.nlp(text)
            # chunks, entities, tokens = self.chunking(doc, entities)
            # res.append([i, text, tokens, chunks, entities])

            sentences, sps, entities = self.my_split(text, self.sentenc_sp, entities)
            for j, sent in enumerate(sentences):
                doc = self.nlp(sent)
                chunks, ents, tokens = self.chunking(sent, doc, entities[j])                   
                res.append(["{}_{}".format(i, j), sent + sps[j], tokens, chunks, ents])

        return pd.DataFrame(res, columns=cols)
