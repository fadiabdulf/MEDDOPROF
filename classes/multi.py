import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from collections import OrderedDict
import re
import os
import zipfile
from multiprocessing import Pool, Process, Manager
import threading


class MyThread (threading.Thread):
    def __init__(self, thread_id, fun, **kwargs):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.fun = fun
        self.kwargs = kwargs
        self.res = None
        
    def run(self):
        self.kwargs['thread_id'] = self.thread_id
        self.res = self.fun(**self.kwargs)

class Extractor(object):
    """
    """
    def __init__(self, df, out_path, multi_processing=False, multi_threading=False, process_count=1):
        self.df = df
        self.out_path = out_path
        self.multi_processing = multi_processing
        self.p_count = process_count
        self.multi_threading = multi_threading
        self.texts = None
        self.labels = None
        self.ZipFile = None
        rule = re.compile('\[\*\*((.+?))\*\*]', re.IGNORECASE)
        date_rule = re.compile('^\d{4}-\d{1,}-\d{1,}$', re.IGNORECASE)
        MonthYear_rule = re.compile('^\d{1,}-?\d{4,}$', re.IGNORECASE)
        Number_rule = re.compile('^\d{1,}$', re.IGNORECASE)
        year_rule = re.compile('^\d{4}$', re.IGNORECASE)
        interval_rule = re.compile('^\d+\-\d+$', re.IGNORECASE)
        other_rule = re.compile('(.+?)(\(|\d{1,}?)', re.IGNORECASE)
        self.rules = {'general': rule, 'date': date_rule, 'month_year': MonthYear_rule, 'number': Number_rule,'year': year_rule, 'interval': interval_rule, 'other': other_rule}

    def _threading(self, data, fun):
        chunk_size = len(data) // self.p_count
        threads = []
        for i in range(0, self.p_count):
            start = i * chunk_size
            end = (start + chunk_size) if i < (self.p_count - 1) else len(data)
            d = data[start:end]
            threads.append(MyThread(i, fun, data=d))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return threads

    # def _processing(self, data, fun):
    #     chunk_size = len(data) // self.p_count
    #     ps = []
    #     manager = Manager()
    #     texts, labels = manager.dict(), manager.dict()
    #     for i in range(0, self.p_count):
    #         start = i * chunk_size
    #         end = (start + chunk_size) if i < (self.p_count - 1) else len(data)
    #         d = data[start:end]
    #         ps.append(Process(target=fun, args=(d, texts, labels,)))
    #         ps[-1].start()

    #     for p in ps:
    #         p.join()
    #     return texts, labels

    def _extract_from_row_threading(self, data, thread_id):
        texts = OrderedDict()
        labels = OrderedDict()
        for row in tqdm(data, total=len(data), miniters=1.0):
            file_id, tags = self._extract_from_row(row)
            texts[file_id] = row['TEXT']
            labels[file_id] = tags
        return texts, labels

    # def _extract_from_row_processing(self, data, *args):
    #     texts, labels = args[0], args[1]
    #     for row in tqdm(data, total=len(data), miniters=1.0):
    #         file_id, tags = self._extract_from_row(row)
    #         texts[file_id] = row['TEXT']
    #         labels[file_id] = tags
    
    def _extract_from_row(self, row):
        file_id = row['ROW_ID']
        tags = []
        for m in re.finditer(self.rules['general'], row['TEXT']):
            tags.append([m.span()[0], m.span()[1], ''])
            tag_type = ''
            if re.search(self.rules['date'], m.group(1)):
                tag_type = 'Date'
            elif re.search(self.rules['year'], m.group(1)):
                tag_type = 'Year'
            elif re.search(self.rules['month_year'], m.group(1)):
                tag_type = 'Month_Year'
            elif re.search(self.rules['number'], m.group(1)):
                tag_type = 'Number'
            elif re.search(self.rules['interval'], m.group(1)):
                tag_type = 'Interval'
            elif re.search(r'\d', m.group(1)):
                try:
                    m2 = re.match(self.rules['other'], m.group(1))
                    tag_type = m2.group(1)
                except:
                    # print(m)
                    tag_type = 'Other'
            else:
                tag_type = m.group(1)
            if not tag_type.strip():
                tag_type = 'Other'
            tags[-1][-1] = '_'.join(tag_type.upper().split())
        return file_id, tags

    def extract(self):
        df = self.df
        texts = OrderedDict()
        labels = OrderedDict()
        if self.multi_threading:
            data = self.df.to_dict('records')
            threads = self._threading(data, self._extract_from_row_threading)
            for thread in threads:
                txts, lbls = thread.res
                texts.update(txts)
                labels.update(lbls)
        elif self.multi_processing:
            # data = self.df.to_dict('records')
            # texts, labels = self._processing(data, self._extract_from_row_processing)
            with Pool(self.p_count) as p:
                # rows = [row for i, row in df.iterrows()]
                rows = df.to_dict('records')
                tags = p.map(self._extract_from_row, tqdm(rows, total=len(df)))
                labels = dict(tags)
                texts = dict(zip(df['ROW_ID'], df['TEXT']))
        else:
            for i, row in tqdm(df.iterrows(), total=len(df)):
                file_id, tags = self._extract_from_row(row)
                texts[file_id] = row['TEXT']
                labels[file_id] = tags
        self.texts = texts
        self.labels = labels

    def _save_content_threading(self, data, thread_id):
        ZipFile = zipfile.ZipFile(os.path.join(self.out_path, "brat_{}.zip".format(thread_id)), "w", zipfile.ZIP_DEFLATED) if self.zip_file else None
        for params in tqdm(data, total=len(data), miniters=1.0):
            params.append(ZipFile)
            self._save_content(*params)
        if not ZipFile is None:
            ZipFile.close()
    
    def _save_content(self, file_id, txt, ann_content, ZipFile):
        if not ZipFile is None:
            ZipFile.writestr('{}.txt'.format(file_id), txt)
        else:
            with open(os.path.join(self.out_path, '{}.txt'.format(file_id)), 'w+', encoding='UTF-8') as f:
                f.write(txt)
        if not ZipFile is None:
            ZipFile.writestr('{}.ann'.format(file_id), ann_content)
        else:
            with open(os.path.join(self.out_path, '{}.ann'.format(file_id)), 'w+', encoding='UTF-8') as f:
                f.write(ann_content)
            
    def save_in_brat_format(self, zip_file=False, multi_zip=1, multi_processing=None, multi_threading=None, process_count=None):
        if not multi_processing is None:
            self.multi_processing = multi_processing
        if not multi_threading is None:
            self.multi_threading = multi_threading
        if not process_count is None:
            self.p_count = process_count
        self.zip_file = zip_file
        if self.texts is None:
            raise Exception("You have to run 'extract' function first." )
        columns = ['name', 'link', 'filter']
        data = []
        ZipFile = None
        ZipFiles = []
        chunk_size = len(self.texts)
        ZipFile = zipfile.ZipFile(os.path.join(self.out_path, "brat_0.zip"), "w", zipfile.ZIP_DEFLATED) if zip_file else None
        ZipFiles.append(ZipFile)
        chunk_size = len(self.texts) // multi_zip
        prepared_for_multi = []
        for i, key in tqdm(enumerate(self.texts), total=len(self.texts), miniters=5.0):
            txt, tag = self.texts[key], self.labels[key]
            file_id = "patient_{}".format(key)
            ann_content = ''
            for i, (start, end, t) in enumerate(tag):
                ann_content += "T{}\t{}\t{}\n".format(i, "{} {} {}".format(t, start, end), txt[start:end]) 
            if not self.multi_processing and not self.multi_threading:
                if multi_zip > 1:
                    zip_file_id = min(i // chunk_size, chunk_size-1)
                    if zip_file_id >= len(ZipFiles):
                        # print("start new zip file i={}, zip_id={}".format(i, zip_file_id))
                        ZipFile = zipfile.ZipFile(os.path.join(self.out_path, "brat_{}.zip".format(zip_file_id)), "w", zipfile.ZIP_DEFLATED)
                        ZipFiles[-1].close()
                        ZipFiles.append(ZipFile)
                self._save_content(file_id, txt, ann_content, ZipFile)
            else:
                prepared_for_multi.append([file_id, txt, ann_content])
            data.append([file_id, '', False])
        if self.multi_threading or self.multi_processing:
            self._threading(prepared_for_multi, self._save_content_threading)
        # elif self.multi_processing:
        #     with Pool(self.p_count) as p:
        #         p.map(self._save_content, tqdm(prepared_for_multi))
        if not ZipFile is None:
            ZipFile.close()
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(os.path.join(self.out_path, 'filters.csv'), index=False)
        return df