FROM python:3.9

RUN apt-get update -y && apt-get install -y libsuitesparse-dev

RUN mkdir /app

COPY *.py /app/

COPY requirements.txt /opt/requirements.txt

RUN pip3 install -r /opt/requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /app