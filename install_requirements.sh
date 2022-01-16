apt update -y
apt install -y git 
apt install -y graphviz
apt install -y docker 
apt install -y docker.io

pip install --update pip
pip install -r requirements.txt

python -c " import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_md
python -m spacy download xx_sent_ud_sm