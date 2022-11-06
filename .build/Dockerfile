FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

ARG USER_NAME
ARG UID
ARG GID
ARG OCAML_COMPILER
USER root

RUN apt-get update && apt-get install -y sudo git inotify-tools build-essential libssl-dev liblapacke-dev libopenblas-dev pkg-config zlib1g-dev libffi-dev libgmp-dev libpcre3-dev tmux fish software-properties-common wget netbase

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3

RUN add-apt-repository ppa:avsm/ppa
RUN apt update
RUN apt install -y opam
ENV USER_NAME=${USER_NAME}
ENV OCAML_COMPILER=${OCAML_COMPILER}

RUN echo "${USER_NAME} ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}
RUN chmod 440 /etc/sudoers.d/${USER_NAME}
RUN chown root:root /etc/sudoers.d/${USER_NAME}
RUN groupadd -g ${GID} ${USER_NAME}
RUN useradd -m ${USER_NAME} --uid=${UID} --gid=${GID} -s /bin/bash
# RUN adduser --uid ${UID} --disabled-password --gecos '' ${USER_NAME}
RUN passwd -l ${USER_NAME}
RUN chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}
USER ${USER_NAME}
ENV HOME=/home/${USER_NAME}

WORKDIR ${HOME}
# Opam installs

RUN opam init -a -y -c ${OCAML_COMPILER} --disable-sandboxing
RUN wget --no-check-certificate https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcu117.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117.zip
ENV LIBTORCH=${HOME}/libtorch
RUN rm -rf libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117.zip

RUN git clone https://github.com/LaurentMazare/ocaml-torch.git
RUN cd ocaml-torch && opam install -y .
RUN opam install -y tls lwt_ssl cohttp-lwt-unix yojson merlin dune ppx_yojson_conv uutf utop npy
WORKDIR ${HOME}