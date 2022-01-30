NAME=tracking

WASI_SDK_PATH = /opt/wasi-sdk-14.0
WAMR_SDK_PATH = /opt/wamr-sdk

#CC = wasicc
CC = $(WASI_SDK_PATH)/bin/clang --sysroot=$(WASI_SDK_PATH)/share/wasi-sysroot
#CXX = wasic++
CXX = $(WASI_SDK_PATH)/bin/clang++ --sysroot=$(WASI_SDK_PATH)/share/wasi-sysroot

LD = $(WASI_SDK_PATH)/bin/wasm-ld

SRCDIR=darknet/src/
OBJDIR=./obj/

#MAINSRC = darknet/example/detector.c

CSRC = gemm.c utils.c cuda.c deconvolutional_layer.c convolutional_layer.c 
CSRC += list.c image.c activations.c im2col.c col2im.c 
CSRC += blas.c crop_layer.c dropout_layer.c maxpool_layer.c 
CSRC += softmax_layer.c data.c matrix.c network.c 
CSRC += connected_layer.c cost_layer.c parser.c option_list.c 
CSRC += detection_layer.c route_layer.c upsample_layer.c box.c 
CSRC += normalization_layer.c avgpool_layer.c layer.c local_layer.c 
CSRC += shortcut_layer.c logistic_layer.c activation_layer.c 
CSRC += rnn_layer.c gru_layer.c crnn_layer.c demo.c batchnorm_layer.c 
CSRC += region_layer.c reorg_layer.c tree.c  lstm_layer.c 
CSRC += l2norm_layer.c yolo_layer.c iseg_layer.c detector.c
CSRCS = $(addprefix $(SRCDIR), $(CSRC))

CPPSRC = image_opencv.cpp
CPPSRCS = $(addprefix $(SRCDIR), $(CPPSRC))

#$SRC = trackers/kalman.cpp
#SRC = trackers/kcf.cpp 
#SRC = trackers/hungarian/hungarian.cpp

#SYSROOT = /opt/wasi-sdk-14.0/share/wasi-sysroot

CSRCS += $(MAINSRC)
COBJ = $(CSRCS:.c=.o)
COBJ += $(CPPSRCS:.cpp=.o)

COBJS = $(addprefix $(OBJDIR), $(COBJ))

CDEPS = $(CSRCS:.c=.d)

CFLAGS = -pthread -Ofast \
	     -g -Wall -Wextra -Wformat -Wformat-security

LD_FLAGS =  -Wl,--strip-all \
            -Wl,--no-entry \
	    -Wl,--initial-memory=6553600 \
            -Wl,--max-memory=1073741824 \
	    -z stack-size=655360 \
	    -Wl,--shared-memory,--no-check-features \
	    -Wl,--allow-undefined      \
	    -Wl,--export=__heap_base,--export=__data_end,--export=load_network,--export=set_batch_network \
	    -Wl,--export=load_image_color,--export=resize_image,--export=network_predict,--export=get_network_boxes,--export=network_predict_image \
	    -Wl,--stack-first

#CFLAGS = -O3                 \
    -fno-exceptions            \
    -Wl,--strip-all            \
    -z stack-size=8192         \
    -Wl,--initial-memory=65536 \
    -mexec-model=reactor       \
    -Wl,--no-entry,--allow-undefined \
    -Wl,--export=__heap_base -Wl,--export=__data_end \
    -Wl,--export=tracker_new

INCLUDE = -I$(WASI_SDK_PATH)/share/wasi-sysroot/include
INCLUDE += -I./darknet/include

WASM=$(NAME).wasm
HEADER=$(NAME).wasm.h
#EMCC_FLAGS=-s ERROR_ON_UNDEFINED_SYMBOLS=0 -s EXCEPTION_DEBUG=1 -s TOTAL_MEMORY=1024MB -s ALLOW_MEMORY_GROWTH=1 -s ASSERTIONS=1 -s USE_PTHREADS=0 -s LLD_REPORT_UNDEFINED -s DEMANGLE_SUPPORT=1
EMCC_FLAGS=

all: obj $(WASM)

clean:
	rm -rf obj
	rm -f $(HEADER) $(WASM)

.PHONY: all clean

$(WASM): $(COBJS)
	$(CC) $(LD_FLAGS) $^ -o $@ 

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

obj:
	mkdir -p obj/$(SRCDIR)

$(HEADER): $(WASM)
	xxd -i $< >$@
