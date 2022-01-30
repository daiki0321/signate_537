# signate_537

## Install
Need to downlonad volov3-tiny.weights file.

~~~
wget https://pjreddie.com/media/files/yolov3-tiny.weights
~~~


## Build tracking.wasm
### When you build in host enfironmebt

~~~
$ cd wasm_module

# In order to enable WASI_GEMM
$ make WASI_GEMM_ENABLE=1
~~~

### When you build on docker

~~~
$ make wasm_module
~~~

## Build main include wasi

~~~
$ cd src

$ make

# In order to enable RISC-V
$ make WASI_GEMM_RISC_V=1
~~~

### When you build on docker

~~~
$ make wamr_main
~~~
