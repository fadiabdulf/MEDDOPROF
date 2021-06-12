# 1. Introduction

Scripts to compute MEDDOPROF evaluation metrics.

Written in Python 3.8

Output is printed in terminal.

# 2. Requirements

+ Python3
+ pandas

To install them: 
```
pip install -r requirements.txt
```


# 3. Execution
+ MEDDOPROF-NER

```
cd src  
python main.py -g ../gs-data/ner/ -p ../toy-data/ner/ -s ner
```

+ MEDDOPROF-CLASS

```
cd src
python main.py -g ../gs-data/class/ -p ../toy-data/class -s class
```

+ MEDDOPROF-NORM

```
cd src
python main.py -g ../gs-data/gs-norm.tsv -p ../toy-data/pred-norm.tsv -c ../meddoprof_valid_codes.tsv.tsv -s norm
```

# 4. Other interesting stuff:
### Metrics
For all subtasks, the relevant metrics are precision, recall and f1-score. The latter will be used to decide the award winners.

### Script Arguments
+ ```-g/--gs_path```: path to directory with Gold Standard .ann files (if we are in subtask NER or CLASS) or path to Gold Standard TSV file (if we are in subtask NORM)
+ ```-p/--pred_path```: path to directory with Prediction .ann files (if we are in subtask NER or CLASS) or path to Prediction TSV file (if we are in subtask NORM)
+ ```-c/--valid_codes_path```: path to TSV file with valid codes (provided here). Codes not included in this TSV will not be used for metrics computation.
+ ```-s/--subtask```: subtask name (```ner```, ```class```, or ```norm```).

### Examples: 
+ MEDDOPROF-NER

```
$ cd src
$ python main.py -g ../gs-data/ner/ -p ../toy-data/meddoprof-ner/ -s ner

-----------------------------------------------------
Clinical case name			Precision
-----------------------------------------------------
32631645_ES.ann		0.667
-----------------------------------------------------
caso_clinico_alergologia1.ann		0.75
-----------------------------------------------------

-----------------------------------------------------
Clinical case name			Recall
-----------------------------------------------------
32631645_ES.ann		0.5
-----------------------------------------------------
caso_clinico_alergologia1.ann		0.75
-----------------------------------------------------

-----------------------------------------------------
Clinical case name			F-score
-----------------------------------------------------
32631645_ES.ann		0.571
-----------------------------------------------------
caso_clinico_alergologia1.ann		0.75
-----------------------------------------------------

_____________________________________________________
Micro-average metrics
_____________________________________________________

Micro-average precision = 0.714


Micro-average recall = 0.625


Micro-average F-score = 0.667
```

+ MEDDOPROF-CLASS

```
$ cd src
$ python main.py -g ../gs-data/class/ -p ../toy-data/meddoprof-class/ -s class

-----------------------------------------------------
Clinical case name			Precision
-----------------------------------------------------
32631645_ES.ann		0.333
-----------------------------------------------------
caso_clinico_alergologia1.ann		1.0
-----------------------------------------------------

-----------------------------------------------------
Clinical case name			Recall
-----------------------------------------------------
32631645_ES.ann		1.0
-----------------------------------------------------
caso_clinico_alergologia1.ann		1.0
-----------------------------------------------------

-----------------------------------------------------
Clinical case name			F-score
-----------------------------------------------------
32631645_ES.ann		0.5
-----------------------------------------------------
caso_clinico_alergologia1.ann		1.0
-----------------------------------------------------

_____________________________________________________
Micro-average metrics
_____________________________________________________

Micro-average precision = 0.714


Micro-average recall = 1.0


Micro-average F-score = 0.833
```

+ MEDDOPROF-NORM

```
$ cd src
$ python main.py -g ../gs-data/meddo-norm-gs.csv -p ../toy-data/meddoprof-norm/1-systemDL.tsv -s norm -c ../meddoprof_valid_codes.tsv

-----------------------------------------------------
Clinical case name			Precision
-----------------------------------------------------
32631645_ES		0.0
-----------------------------------------------------
caso_clinico_alergologia1		0.25
-----------------------------------------------------

-----------------------------------------------------
Clinical case name			Recall
-----------------------------------------------------
32631645_ES		0.0
-----------------------------------------------------
caso_clinico_alergologia1		0.25
-----------------------------------------------------

-----------------------------------------------------
Clinical case name			F-score
-----------------------------------------------------
32631645_ES		nan
-----------------------------------------------------
caso_clinico_alergologia1		0.25
-----------------------------------------------------

_____________________________________________________
Micro-average metrics
_____________________________________________________

Micro-average precision = 0.143


Micro-average recall = 0.125


Micro-average F-score = 0.133
```

