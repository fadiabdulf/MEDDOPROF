apt update -y
apt install -y git 
apt install -y graphviz
apt install -y docker 
apt install -y docker.io

pip install --update pip
pip install -r requirements.txt

python -c " import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python -m spacy download es_core_news_sm
python -m spacy download xx_ent_wiki_sm