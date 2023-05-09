FROM python:3.8-buster

ARG UID
ARG GID
ARG GROUPNAME

ARG USERNAME=jovyan
ARG PASSWORD=jovyan

RUN apt-get update && apt-get install -y sudo
#: グループ名重複の場合はユーザー名で代替
RUN groupadd -g $GID -o $GROUPNAME || groupadd -g $GID -o $USERNAME
RUN useradd -m -u $UID -g $GID $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd
RUN usermod -aG sudo $USERNAME

USER $USERNAME
WORKDIR /project

ENV PATH $PATH:/home/jovyan/.local/bin
ENV PYTHONPATH $PYTHONPATH:/project

RUN pip install poetry==1.1.13
RUN poetry config virtualenvs.create false
