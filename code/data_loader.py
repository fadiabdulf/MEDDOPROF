from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as BS
from collections import OrderedDict
from tqdm.auto import tqdm, trange
import json
import zipfile
import datatable as dt
import ast

class BaseDataLoader(object):
    """
    input_dir: the input directory.
    output_dir: the output directory.
    """
    def __init__(self, input_dir, filter, **kwargs):
        self.input_dir = input_dir
        self.filter = filter
        self.filter_path = os.path.join(self.input_dir, self.filter)
        self.df = None

    def read(self, path, file):
        file_path = os.path.join(path, file)
        with open(file_path, 'rb') as f:
            blines = f.readlines()
        lines = [bline.decode("utf-8") for bline in blines]
        content = ''.join(lines)
        return content

    @abstractmethod
    def native_load(self):
        """
        """

    def load(self, refresh=False):
        if self.df is None or refresh:
            data = []
            columns = ['id', 'text', 'main_entity', 'entities']
            available = self.available_data_sets()
            for file, txt, main_ent, ents in tqdm(self.native_load(), total=len(available), desc="Loading ..."):
                data.append([file, txt, main_ent, ents])
            df = pd.DataFrame(data, columns=columns)
            self.df = df
        return self.df


    def available_data_sets(self):
        """
        """
        df = pd.read_csv(self.filter_path)
        columns = ['file_name', 'labeled']
        df2 = pd.DataFrame()
        df2[columns] = df[['name', 'filter']]
        return df2


class DataLoader(BaseDataLoader):
    """
    """
    def __init__(self, loader_type='xml', **kwargs):
        supported_loaders = {'txt': TXTDataLoader(**kwargs), 
                             'xml': XMLDataLoader(**kwargs),
                             'json': JSONDataLoader(**kwargs),
                             'brat': BRATDataLoader(**kwargs),
                             'csv': CSVDataLoader(**kwargs),
                             }
        self.loader_type = loader_type.lower()
        self.loader = supported_loaders[self.loader_type]

    def load(self, **kwargs):
        return self.loader.load(**kwargs)

    def native_load(self, **kwargs):
        return self.loader.native_load(**kwargs)

    def available_data_sets(self, **kwargs):
        return self.loader.available_data_sets(**kwargs)


class TXTDataLoader(BaseDataLoader):
    """
    """
    def __init__(self, input_dir, filter, **kwargs):
        super(TXTDataLoader, self).__init__(input_dir=input_dir, filter=filter)
        self.loader_type = 'txt'

    def read(self, path, file):
        return super(TXTDataLoader, self).read(path, file)

    def native_load(self):
        """
        load text: load text file from the directory 'input_dir'.
        """
        splitter = '\n<tags>\n'
        file_names = [f for f in os.listdir(self.input_dir) if f.endswith('.txt')]
        filt = pd.read_csv(self.filter_path, index_col='name')
        filt['filter'] = filt['filter'].astype(bool)
        for file in file_names:
            text = self.read(self.input_dir, file)
            txts = text.split(splitter)
            if filt.loc[file[:-4], 'filter']:
                soup = BS(splitter + txts[1], features='lxml')
                tags = []
                for t in soup.find_all('tag'):
                    tags.append([int(t['offset']), int(t['offset']) + int(t['length']), t['type']])
            else:
                tags = []
            yield file[:-4], txts[0], tags[0] if len(tags) > 0 else None, tags


class XMLDataLoader(BaseDataLoader):
    """
    """
    def __init__(self, input_dir, filter, abstract_tag='abstract', **kwargs):
        super(XMLDataLoader, self).__init__(input_dir=input_dir, filter=filter)
        self.abstract_tag = abstract_tag
        self.loader_type = 'xml'

    def read(self, path, file, parser='html.parser'):
        content = super(XMLDataLoader, self).read(path, file)
        soup = BS(content, parser)
        #soup = BS(text, 'lxml-xml')
        return soup

    def native_load(self):
        """
        load xml: load xml file from the directory 'input_dir'.
        """
        file_names = [f for f in os.listdir(self.input_dir) if f.endswith('.xml')]
        for file in file_names:
            soup = self.read(self.input_dir, file, parser='xml')
            txt = soup.find(self.abstract_tag).text
            tags = []
            if not soup.find('tags') is None:
                for t in soup.find('tags').find_all('tag'):
                    tags.append([int(t['offset']), int(t['offset']) + int(t['length']), t['type']])
            yield file[:-4], txt, tags[0] if len(tags) > 0 else None, tags

class JSONDataLoader(BaseDataLoader):
    """
    """
    def __init__(self, input_dir, filter, json_file='file.json1', **kwargs):
        super(JSONDataLoader, self).__init__(input_dir=input_dir, filter=filter)
        self.json_file = json_file
        self.loader_type = 'json'

    def read(self, path, file):
        content = super(JSONDataLoader, self).read(path, file)
        return json.load(content)

    def native_load(self):
        """
        load json: load json file from the directory 'input_dir'.
        """
        json_content = self.read(self.input_dir, self.json_file)
        for item in json_content:
            id = row['id']
            txt = item['text']
            tags = item['labels']
            yield id, txt, tags[0] if len(tags) > 0 else None, tags

class BRATDataLoader(BaseDataLoader):
    """
    """
    def __init__(self, input_dir, filter, compressed=False, **kwargs):
        super(BRATDataLoader, self).__init__(input_dir=input_dir, filter=filter)
        self.compressed = compressed
        self.loader_type = 'brat'

    def read(self, path, file, zfile=None):
        if self.compressed:
            ifile = zfile.open(file)
            blines = ifile.readlines()
            lines = [bline.decode("utf-8") for bline in blines]
            content = ''.join(lines)
            return content
        else:
            return super(BRATDataLoader, self).read(path, file)

    def native_load(self):
        """
        load brat: load brat file from the directory 'input_dir'.
        """
        if self.compressed:
            zip_file_names = [f for f in os.listdir(self.input_dir) if f.endswith('.zip')]
            self.zfiles = [zipfile.ZipFile(os.path.join(self.input_dir, file)) for file in tqdm(zip_file_names, desc="Load zip files ...")]
            all_file_names = {}
            for zip_file in tqdm(self.zfiles, desc="Process zip files ..."):
                file_names = {}
                for f in zip_file.infolist():
                    if not f.filename[:-4] in file_names:
                        file_names[f.filename[:-4]] = []
                    file_names[f.filename[:-4]].append(f)
                for key in file_names:
                    file_names[key] = sorted(file_names[key], key=lambda x: x.filename[-4:])
                    file_names[key].append(zip_file)
                all_file_names.update(file_names)
            file_names = all_file_names
        else:
            file_names = [f[:-4] for f in os.listdir(self.input_dir) if f.endswith('.txt')]
        filt = pd.read_csv(self.filter_path, index_col='name')
        filt['filter'] = filt['filter'].astype(bool)
        for file in file_names:
            if self.compressed:
                text = self.read(self.input_dir, file_names[file][1], file_names[file][2])
                labels = self.read(self.input_dir, file_names[file][0], file_names[file][2])
            else:
                text = self.read(self.input_dir, file + '.txt')
                labels = self.read(self.input_dir, file + '.ann')
            tags = []
            for t in labels.split('\n'):
                if t.strip():
                    ts = t.split('\t')
                    ts[1] = ts[1].split()
                    tag_type, start, end = ts[1][0], int(ts[1][1]), int(ts[1][2])
                    tags.append([start, end, tag_type])
            yield file, text, tags[0] if len(tags) > 0 else None, tags


class CSVDataLoader(BaseDataLoader):
    """
    """
    def __init__(self, input_dir, filter, **kwargs):
        super(CSVDataLoader, self).__init__(input_dir=input_dir, filter=filter)
        self.loader_type = 'csv'

    def read(self, path, file):
        df = pd.read_csv(path + file)
        # df['entities'] = df['entities'].apply(ast.literal_eval)
        # df['main_entity'] = df['main_entity'].apply(lambda x: ast.literal_eval(x) if x == x else None)
        return df

    def native_load(self):
        """
        load text: load csv file from the directory 'input_dir'.
        """
        df = self.read(self.input_dir, 'mimic-III.csv')
        for _, row in df.iterrows():
            entities = ast.literal_eval(row['entities'])
            yield row['id'], row['text'], None if len(entities) == 0 else entities[0], entities
            

