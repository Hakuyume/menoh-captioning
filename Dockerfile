FROM debian:stretch AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libprotobuf-dev \
    pkg-config \
    protobuf-compiler \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN echo '/usr/local/lib' > /etc/ld.so.conf.d/local.conf \
    && ldconfig
RUN git clone https://github.com/intel/mkl-dnn.git --branch=v0.16 --depth=1 \
    && cd mkl-dnn/scripts \
    && ./prepare_mkl.sh \
    && cd .. \
    && mkdir build \
    && cd build \
    && cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_EXAMPLE=OFF \
    -DWITH_TEST=OFF \
    && make -j $(nproc) \
    && make install
RUN git clone https://github.com/pfnet-research/menoh.git --branch=v1.0.3 --depth=1 \
    && cd menoh \
    && mkdir build \
    && cd build \
    && cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_BENCHMARK=OFF \
    -DENABLE_EXAMPLE=OFF \
    -DENABLE_TEST=OFF \
    -DENABLE_TOOL=OFF \
    && make -j $(nproc) \
    && make install

COPY . menoh-captioning/

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=stable
RUN cd menoh-captioning \
    && sed \
    -e 's#ImageCaptionModel.onnx#/usr/local/share/ImageCaptionModel.onnx#' \
    -e 's#vocab.txt#/usr/local/share/vocab.txt#' \
    -i src/main.rs \
    && PATH=$HOME/.cargo/bin:$PATH \
    PKG_CONFIG_PATH=/usr/local/share/pkgconfig \
    cargo build --release -j $(nproc) \
    && install -m 755 target/release/menoh-captioning /usr/local/bin/

RUN tar -cvf install.tar \
    /etc/ld.so.conf.d/local.conf \
    /usr/local/bin/menoh-captioning \
    /usr/local/lib

FROM debian:stretch AS deploy
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libprotobuf10 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build install.tar .
RUN tar xvf install.tar -C / \
    && rm install.tar \
    && ldconfig
RUN curl -L https://github.com/Hakuyume/menoh-captioning/releases/download/assets/ImageCaptionModel.onnx -o /usr/local/share/ImageCaptionModel.onnx \
    && curl -L https://github.com/Hakuyume/menoh-captioning/releases/download/assets/vocab.txt -o /usr/local/share/vocab.txt \
    && curl -LO https://upload.wikimedia.org/wikipedia/commons/7/79/Trillium_Poncho_cat_dog.jpg
