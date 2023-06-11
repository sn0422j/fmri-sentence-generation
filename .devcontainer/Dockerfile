FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install fonts-noto-cjk
RUN conda install scipy matplotlib seaborn scikit-learn pandas nltk boto3 regex pylint -y
RUN conda install -c conda-forge sacremoses bert_score black isort -y
