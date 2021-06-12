#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:32:08 2021

@author: tonifuc3m
"""

import argparse
import warnings
import pandas as pd
import os

import ann_parsing
import compute_metrics

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line

def parse_arguments():
    '''
    DESCRIPTION: Parse command line arguments
    '''
  
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.add_argument('-g', '--gs_path', required = True, dest = 'gs_path', 
                        help = 'path to GS file')
    parser.add_argument('-p', '--pred_path', required = True, dest = 'pred_path', 
                        help = 'path to predictions file')
    parser.add_argument('-c', '--valid_codes_path', required = False, 
                        default = '../meddoprof_valid_codes.tsv',
                        dest = 'codes_path', help = 'path to valid codes TSV')
    parser.add_argument('-s', '--subtask', required = True, dest = 'subtask',
                        choices=['class', 'ner', 'norm'],
                        help = 'Subtask name')
    
    args = parser.parse_args()
    gs_path = args.gs_path
    pred_path = args.pred_path
    codes_path = args.codes_path
    subtask = args.subtask
    
    return gs_path, pred_path, codes_path, subtask

def main(gs_path, pred_path, subtask=['class', 'ner', 'norm'], codes_path='', g=None, p=None):
    '''
    Load GS and Predictions; format them; compute precision, recall and 
    F1-score and print them.

    Parameters
    ----------
    gs_path : str
        Path to directory with GS in Brat (or to GS in TSV if subtask is norm).
    pred_path : str
        Path to directory with Predicted files in Brat (or to GS in TSV if subtask is norm).
    subtask : str
        Subtask name
    codes_path : str
        Path to TSV with valid codes

    Returns
    -------
    None.

    '''
    
    if subtask=='norm':
        gs = pd.read_csv(gs_path, sep='\t', header=0) if g is None else g
        pred = pd.read_csv(pred_path, sep='\t', header=0) if p is None else p
        
        if pred.shape[0] == 0:
            raise Exception('There are not parsed predicted annotations')
        elif gs.shape[0] == 0:
            raise Exception('There are not parsed Gold Standard annotations')
        if pred.shape[1] != 4:
            raise Exception('Wrong column number in predictions file')
        elif gs.shape[1] != 4:
            raise Exception('Wrong column number in Gold Standard file')
            
        gs.columns = ['clinical_case', 'span', 'offset', 'code']
        pred.columns = ['clinical_case', 'span', 'offset', 'code']
        
        pred['offset'] = pred['offset'].apply(lambda x: x.strip())
        pred['code'] = pred['code'].apply(lambda x: x.strip())
        pred['clinical_case'] = pred['clinical_case'].apply(lambda x: x.strip())
        
    elif subtask in ['class', 'ner']:
        
        if subtask=='class':
            labels = ['SANITARIO', 'PACIENTE', 'FAMILIAR','OTROS']
        elif subtask=='ner':
            labels = ['ACTIVIDAD', 'PROFESION', 'SITUACION_LABORAL']

        gs = ann_parsing.main(gs_path, labels, with_notes=False) if g is None else g
        pred = ann_parsing.main(pred_path, labels, with_notes=False) if p is None else p
        
        if pred.shape[0] == 0:
            raise Exception('There are not parsed predicted annotations')
        elif gs.shape[0] == 0:
            raise Exception('There are not parsed Gold Standard annotations')
                   
        gs.columns = ['clinical_case', 'mark', 'label', 'offset', 'span', 
                      'start_pos_gs', 'end_pos_gs']
        pred.columns = ['clinical_case', 'mark', 'label', 'offset', 'span',
                      'start_pos_pred', 'end_pos_pred']   

    # Remove predictions for files not in Gold Standard
    if subtask in ['ner', 'class']:
        doc_list_gs = list(filter(lambda x: x[-4:] == '.ann', os.listdir(gs_path))) 
    elif subtask == 'norm':
        doc_list_gs = list(set(gs['clinical_case'].tolist()))
    pred_gs_subset = pred.loc[pred['clinical_case'].isin(doc_list_gs),:].copy()
    
    if pred_gs_subset.shape[0] == 0:
        raise Exception('There are not valid predicted annotations. '+
                        'The only predictions are for files not in Gold Standard')
    
    # Remove predictions for codes not valid
    if subtask=='norm':
        valid_codes = pd.read_csv(codes_path, sep='\t', header=0)['code'].tolist()
        pred_gs_subset = pred_gs_subset.loc[pred['code'].isin(valid_codes),:].copy()
        
    if pred_gs_subset.shape[0] == 0:
        raise Exception('There are not valid predicted annotations. '+
                        'The only predictions contain invalid codes')
        
    # Compute metrics
    P_per_cc, P, R_per_cc, R, F1_per_cc, F1 = \
        compute_metrics.main(gs, pred_gs_subset, doc_list_gs, subtask=subtask)
        
    ###### Show results ######  
    # print('\n-----------------------------------------------------')
    # print('Clinical case name\t\t\tPrecision')
    # print('-----------------------------------------------------')
    # for index, val in P_per_cc.items():
    #     print(str(index) + '\t\t' + str(round(val, 3)))
    #     print('-----------------------------------------------------')       
    
    # print('\n-----------------------------------------------------')
    # print('Clinical case name\t\t\tRecall')
    # print('-----------------------------------------------------')
    # for index, val in R_per_cc.items():
    #     print(str(index) + '\t\t' + str(round(val, 3)))
    #     print('-----------------------------------------------------')    
    
    # print('\n-----------------------------------------------------')
    # print('Clinical case name\t\t\tF-score')
    # print('-----------------------------------------------------')
    # for index, val in F1_per_cc.items():
    #     print(str(index) + '\t\t' + str(round(val, 3)))
    #     print('-----------------------------------------------------')
        
    print('_____________________________________________________')
    print('Micro-average metrics [{}]'.format(subtask))
    print('_____________________________________________________')
    print('Micro-average precision = {}'.format(round(P, 3)))
    print('   Micro-average recall = {}'.format(round(R, 3)))
    print('  Micro-average F-score = {}'.format(round(F1, 3)))
    
    #print('{}|{}|{}|{}'.format(pred_path,round(P, 3),round(R, 3),round(F1, 3)))
    return P, R, F1
    
    
if __name__ == '__main__':
    
    gs_path, pred_path, codes_path, subtask = parse_arguments()
    
    if os.path.exists(gs_path)==False:
        raise Exception('Gold Standard path does not exist')
    if os.path.exists(pred_path)==False:
        raise Exception('Predictions path does not exist')
    if os.path.exists(codes_path)==False:
        raise Exception('Codes path does not exist')
    if subtask not in ['ner', 'class', 'norm']:
        raise Exception('Error! Subtask name does not exist')
        
    main(gs_path, pred_path, subtask=subtask, codes_path=codes_path)
