FROM ubuntu:20.04 as tools
ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y make git cmake wget libeigen3-dev unzip libboost-all-dev \
	clang-11 clang-tools-11 \
	clang++-11

RUN update-alternatives --install /usr/bin/cc cc /usr/bin/clang-11 50
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-11 50
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-11 50
RUN apt-get remove -y gcc

WORKDIR /root/src
RUN git clone --depth=1 https://github.com/bytecodealliance/wasm-micro-runtime

ARG CC
ARG ARCH=AARCH64
WORKDIR /root/src/wasm-micro-runtime/product-mini/platforms/linux/build
RUN cmake \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DWAMR_BUILD_TARGET=${ARCH} \
    -DWAMR_BUILD_INTERP=1 \
    -DWAMR_BUILD_FAST_INTERP=1 \
    -DWAMR_BUILD_JIT=0 \
    -DWAMR_BUILD_AOT=1 \
    -DWAMR_BUILD_LIBC_WASI=1 \
    -DWAMR_BUILD_LIBC_BUILTIN=1 \
    -DWAMR_BUILD_LIBC_UVWASI=0 \
    -DWAMR_BUILD_MULTI_MODULE=0 \
    -DWAMR_BUILD_MINI_LOADER=0 \
    -DWAMR_BUILD_SHARED_MEMORY=1 \
    -DWAMR_BUILD_THREAD_MGR=1 \
    -DWAMR_BUILD_LIB_PTHREAD=1 \
    -DWAMR_BUILD_BULK_MEMORY=1 \
    -DWAMR_APP_THREAD_STACK_SIZE_MAX=83886080 \
    -DWAMR_BUILD_MEMORY_PROFILING=1 \
    ..
RUN make
RUN install -m 755 /root/src/wasm-micro-runtime/product-mini/platforms/linux/build/libvmlib.a /usr/local/lib
RUN install -m 644 /root/src/wasm-micro-runtime/core/iwasm/include/lib_export.h /usr/local/include
RUN install -m 644 /root/src/wasm-micro-runtime/core/iwasm/include/wasm_export.h /usr/local/include

WORKDIR /root/src/
# install opencv
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.3.zip
RUN unzip opencv.zip

RUN mkdir build

WORKDIR /root/src/build
RUN cmake ../opencv-3.4.3 -DCMAKE_INSTALL_PREFIX=/root/src/build && make -j4 && make install

WORKDIR /root/src
RUN git clone https://github.com/leggedrobotics/tensorflow-cpp

WORKDIR /root/src/tensorflow-cpp/eigen
RUN ./install.sh

WORKDIR /root/src/tensorflow-cpp/tensorflow
RUN mkdir build

WORKDIR /root/src/tensorflow-cpp/tensorflow/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/root/src/build -DCMAKE_BUILD_TYPE=Release ..
RUN make install -j

RUN cp /root/src/build/lib/libtensorflow_framework.so.1 /root/src/build/lib/libtensorflow_framework.so

WORKDIR /root/src
COPY ./src /root/src/main

WORKDIR /root/src/main
RUN make clean && make



