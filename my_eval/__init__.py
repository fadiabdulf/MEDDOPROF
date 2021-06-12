import sys 
sys.path.append('./meddoprof-evaluation-library-main/')

from .src import main as ev

def my_eval(subtask, gs, pred, doc_list=None):
    if doc_list is None:
        doc_list = list(dict(gs['clinical_case'].value_counts()))
    P, R, F1 = ev.main(None, None, subtask=subtask, g=gs, p=pred, doc_list=doc_list)
    return F1
    

if __name__ == '__main__':
 import pandas as pd
 import argparse
 parser = argparse.ArgumentParser(description='process user given parameters')
 parser.add_argument('-g', '--gs_path', required = True, dest = 'gs_path', 
                    help = 'path to GS file')
 parser.add_argument('-p', '--pred_path', required = True, dest = 'pred_path', 
                help = 'path to predictions file')
 t1, t2 = None, None
 try:
    args = parser.parse_args()
    print(args)
    t1, t2 = args.gs_path, args.pred_path
 except Exception as ex:
    print(ex)
 subtask = 'norm'
 if t1 is None:
     
     columns = ['clinical_case', 'mark', 'label', 'offset', 'span', 'start_pos_pred', 'end_pos_pred', 'code']
     data = [['casos_clinicos_profesiones89.ann', 'T0', '2211', '660 666', '2211', '660', '666', '2211'],
             ['casos_clinicos_profesiones89.ann', 'T0', '2211', '660 666', '2211', '660', '666', '2211'],
             ['casos_clinicos_profesiones83.ann', 'T0', '2212', '9810 9820', '2212', '9810', '9820', '2212'],
             ['casos_clinicos_profesiones83.ann', 'T0', '2212', '9953 9967', '2212', '9953', '9967', '2212'],
             ['casos_clinicos_profesiones89.ann', 'T0', '2211', '13 19', '2211', '13', '19', '2211']]
     gs = pd.DataFrame(data, columns=columns)
     data = [['casos_clinicos_profesiones89.ann', 'T0', '2222', '660 666', '2211', '660', '666', '1111'],
             ['casos_clinicos_profesiones89.ann', 'T0', '5412', '1176 1183', '5412', '1176', '1183', '5412'],
             ['casos_clinicos_profesiones83.ann', 'T0', '2212', '9810 9820', '2212', '9810', '9820', '2212'],
             ['casos_clinicos_profesiones83.ann', 'T0', '2212', '9810 9820', '2212', '9810', '9820', '2212'],
             ['casos_clinicos_profesiones89.ann', 'T0', '2211', '13 19', '2211', '13', '19', '2211']]
     pred = pd.DataFrame(data, columns=columns)
 else:
    gs = pd.read_csv(t1, sep='\t')
    pred = pd.read_csv(t2, sep='\t')
 
 my_eval(subtask, gs, pred)