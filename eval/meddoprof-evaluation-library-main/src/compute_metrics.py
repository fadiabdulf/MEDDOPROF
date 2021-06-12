#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:32:08 2021

@author: tonifuc3m
"""

import pandas as pd
import warnings

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)
warnings.formatwarning = warning_on_one_line


def main(gs, pred, doc_list_gs, subtask=['ner','class', 'norm']):
    '''       
    Calculate task Coding metrics:
    
    Two type of metrics are calculated: per document and micro-average.
    It is assumed there are not completely overlapping annotations.
    
    Parameters
    ---------- 
    gs : pandas dataframe
        with the Gold Standard. Columns are those defined in main function.
    pred : pandas dataframe
        with the predictions. Columns are those defined in main function.
    doc_list_gs : list
        list of documents in Gold Standard
    subtask : str
        subtask name
    
    Returns
    -------
    P_per_cc : pandas series
        Precision per clinical case (index contains clinical case names)
    P : float
        Micro-average precision
    R_per_cc : pandas series
        Recall per clinical case (index contains clinical case names)
    R : float
        Micro-average recall
    F1_per_cc : pandas series
        F-score per clinical case (index contains clinical case names)
    F1 : float
        Micro-average F1-score
    '''
    
    if subtask == "norm":
        relevant_columns = ["clinical_case", "offset", "code"]
    elif subtask in ["ner", "class"]:
        relevant_columns = ["clinical_case", "offset"]
        
    # Predicted Positives:
    Pred_Pos_per_cc = \
        pred.drop_duplicates(subset=relevant_columns).\
        groupby("clinical_case")["offset"].count()
    Pred_Pos = pred.drop_duplicates(subset=relevant_columns).shape[0]

    # Gold Standard Positives:
    GS_Pos_per_cc = \
        gs.drop_duplicates(subset=relevant_columns).\
        groupby("clinical_case")["offset"].count()
    GS_Pos = gs.drop_duplicates(subset=relevant_columns).shape[0]
    
    # Eliminate predictions not in GS (prediction needs to be in same clinical
    # case and to have the exact same offset to be considered valid!!!!)
    df_sel = pd.merge(pred, gs, 
                      how="right",
                      on=relevant_columns)
    is_valid = df_sel.apply(lambda x: x.isnull().any()==False, axis=1)
    df_sel = df_sel.assign(is_valid=is_valid.values)   
       
    # True Positives:
    TP_per_cc = (df_sel[df_sel["is_valid"] == True]
                 .groupby("clinical_case")["is_valid"].count())
    TP = df_sel[df_sel["is_valid"] == True].shape[0]
    
    # Add entries for clinical cases that are not in predictions but are present
    # in the GS
    cc_not_predicted = (pred.drop_duplicates(subset=["clinical_case"])
                        .merge(gs.drop_duplicates(subset=["clinical_case"]), 
                              on='clinical_case',
                              how='right', indicator=True)
                        .query('_merge == "right_only"')
                        .drop('_merge', 1))['clinical_case'].to_list()
    for cc in cc_not_predicted:
        TP_per_cc[cc] = 0
        
    # Add TP = 0 in clinical cases where all predictions are wrong
    for doc in doc_list_gs:
        if doc not in TP_per_cc.index.tolist():
            TP_per_cc[doc] = 0
    
    # Remove entries for clinical cases that are not in GS but are present
    # in the predictions
    cc_not_GS = (gs.drop_duplicates(subset=["clinical_case"])
                .merge(pred.drop_duplicates(subset=["clinical_case"]), 
                      on='clinical_case',
                      how='right', indicator=True)
                .query('_merge == "right_only"')
                .drop('_merge', 1))['clinical_case'].to_list()
    Pred_Pos_per_cc = Pred_Pos_per_cc.drop(cc_not_GS)

    # Calculate Final Metrics:
    P_per_cc =  TP_per_cc / Pred_Pos_per_cc
    P = TP / Pred_Pos
    R_per_cc = TP_per_cc / GS_Pos_per_cc
    R = TP / GS_Pos
    F1_per_cc = (2 * P_per_cc * R_per_cc) / (P_per_cc + R_per_cc)
    if (P+R) == 0:
        F1 = 0
        warnings.warn('Global F1 score automatically set to zero to avoid division by zero')
        return P_per_cc, P, R_per_cc, R, F1_per_cc, F1
    F1 = (2 * P * R) / (P + R)
    
    
    if ((any([F1, P, R]) > 1) | any(F1_per_cc>1) | any(P_per_cc>1) | any(R_per_cc>1) ):
        warnings.warn('Metric greater than 1! You have encountered an undetected bug, please, contact antoniomiresc@gmail.com!')
                                            
    return P_per_cc, P, R_per_cc, R, F1_per_cc, F1