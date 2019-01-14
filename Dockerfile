FROM ubuntu:18.04

ENV DOWNLOAD https://github.com/pfnet-research/menoh/releases/download
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-setuptools \
    && curl -LO $DOWNLOAD/v1.1.0/ubuntu1804_mkl-dnn_0.16-1_amd64.deb \
    && curl -LO $DOWNLOAD/v1.1.0/ubuntu1804_menoh_1.1.0-1_amd64.deb \
    && curl -LO $DOWNLOAD/v1.1.0/ubuntu1804_menoh-dev_1.1.0-1_amd64.deb \
    && apt install -y ./*.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm *.deb

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=stable

COPY . menoh-captioning/

RUN cd menoh-captioning \
    && sed \
    -e 's#ImageCaptionModel.onnx#/usr/local/share/ImageCaptionModel.onnx#' \
    -e 's#vocab.txt#/usr/local/share/vocab.txt#' \
    -i src/main.rs \
    && PATH=$HOME/.cargo/bin:$PATH cargo build --release -j $(nproc) \
    && install -m 755 target/release/menoh-captioning /usr/local/bin/

ENV ASSETS https://github.com/Hakuyume/menoh-captioning/releases/download/assets
RUN pip3 install --no-cache-dir onnx-chainer==1.1.1a2 \
    && curl -LO $ASSETS/model.npz \
    && python3 menoh-captioning/convert.py model.npz --out /usr/local/share/ImageCaptionModel.onnx \
    && rm -rf model.npz $HOME/.chainer \
    && curl -L $ASSETS/vocab.txt -o /usr/local/share/vocab.txt

RUN curl -LO https://upload.wikimedia.org/wikipedia/commons/7/79/Trillium_Poncho_cat_dog.jpg
