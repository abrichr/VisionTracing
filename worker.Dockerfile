FROM nvidia/cuda:11.0-runtime-ubuntu18.04 

RUN apt-get update
RUN apt-get install python3.6 -y
RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-pip -y 

ARG FLASK_ENV=production
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . /app

RUN pip3 install --upgrade pip

RUN apt-get update
RUN apt-get install git -y

# cython depends on numpy, and cython-bbox in worker.requirements.txt depends on 
# cython, and since pip install does not follow the line order in installation,
# have to install numpy and cython before the requirements file 
RUN pip3 install numpy==1.19.1
RUN pip3 install --upgrade cython
RUN pip3 install -r worker.requirements.txt &&  python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# This fixes libgl1 error when you try to import OpenCV
RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y 

CMD python3 worker.py
