FROM nvcr.io/nvidia/pytorch:22.10-py3

WORKDIR /workspace
COPY ./requirements.txt /workspace

RUN pip install -r requirements.txt

ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install opencv-python==4.5.5.64
COPY . /workspace/

ENTRYPOINT ["bash", "entrypoint.sh"]