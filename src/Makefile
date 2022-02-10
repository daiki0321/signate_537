NAME=yolo_main

#CC = wasicc
CC = clang
#CXX = wasic++
CXX = clang++

SRCDIR=./
OBJDIR=./obj/

#MAINSRC = darknet/main.c

#CSRC = main.c utils.c predict.c image.c
CSRC = main.c utils.c predict.c

ifdef WASI_GEMM_RISC_V
CSRC += wasi_gemm_fpga_risc_v.c
else
CSRC += wasi_gemm_cpu.c
endif

CSRCS = $(addprefix $(SRCDIR), $(CSRC))

CPPSRC = tracker/src/track.cpp tracker/src/tracker.cpp tracker/src/kalmanfilter.cpp 
CPPSRC += tracker/src/nn_matching.cpp tracker/src/linear_assignment.cpp tracker/src/hungarianoper.cpp tracker/src/model.cpp tracker/src/deepsort.cpp
CPPSRC += tracker/src/tracker_if.cpp
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

CFLAGS = -pthread -O3 -g

ifdef WASI_GEMM_RISC_V
CFLAGS += -DWASI_GEMM_RISC_V
endif

LD_FLAGS = 
LIBS = -L/usr/local/lib -lvmlib -lm -ldl -lpthread -lrt `pkg-config --libs opencv`

#CFLAGS = -O3                 \
    -fno-exceptions            \
    -Wl,--strip-all            \
    -z stack-size=8192         \
    -Wl,--initial-memory=65536 \
    -mexec-model=reactor       \
    -Wl,--no-entry,--allow-undefined \
    -Wl,--export=__heap_base -Wl,--export=__data_end \
    -Wl,--export=tracker_new

INCLUDE = -I./include -I./wamr/include -I./tracker/include -I/usr/include/eigen3 -I/root/src/build/include 

EXEC=$(NAME)

all: obj $(EXEC)

clean:
	rm -rf obj
	rm -f $(EXEC)

.PHONY: all clean

$(EXEC): $(COBJS)
	$(CXX) $(LD_FLAGS) $^ -o $@ $(LIBS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

obj:
	mkdir -p obj/$(SRCDIR)
	mkdir -p obj/tracker/src
