FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update -y

COPY requirements.txt /tmp/requirements.txt
COPY install_requirements.sh /tmp/install_requirements.sh

RUN cd /tmp && /bin/bash /tmp/install_requirements.sh

RUN pip install jupyter -U && pip install jupyterlab==3.2.1
RUN pip install ipywidgets==7.6.5

RUN jupyter nbextension enable --py widgetsnbextension
RUN curl -L -O -k https://deb.nodesource.com/setup_12.x | /bin/bash
RUN apt-get install -y nodejs
RUN node -v
RUN curl -L -O -k https://raw.githubsercontent.com/nvm-sh/nvm/v0.35.3/install.sh | /bin/bash