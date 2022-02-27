#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "wasm_export.h"
#include "utils.h"

#include "darknet.h"


#define STACK_SIZE 8 * 1024
#define HEAP_SIZE 1024 * 1024 * 1024

wasm_exec_env_t g_exec_env = NULL;

#ifdef WASI_GEMM_RISC_V
x_gemm_fpga_risc_v(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#else
x_gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc);
#endif

x_gemm(wasm_exec_env_t exec_env, int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
#ifdef WASI_GEMM_RISC_V
    x_gemm_fpga_risc_v( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
#else
    x_gemm_cpu(TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
#endif
}

/* predict.c */
int yolo_initialize(char *datacfg, char *cfgfile, char *weightfile);
int test_detector(char *filename, char *outfile);
void callback_predict_result(wasm_exec_env_t exec_env, void* dets, int total, int classes, int w, int h);
void print_save_json_result(char* filename);

static wasm_function_inst_t get_function(wasm_module_inst_t module_inst, char* func_name) {

    wasm_function_inst_t func = wasm_runtime_lookup_function(module_inst, func_name, NULL);
    if(!func) {                                       
        printf("The %s wasm function is not found.\n", func_name);
        assert(0);
    }

    return func;
}

int call_wasm_function(wasm_exec_env_t exec_env, char* func_name, int argc, char** argv) {

    wasm_function_inst_t module_inst = get_module_inst(exec_env);
    wasm_function_inst_t func = get_function(module_inst, func_name);

    if (!wasm_runtime_call_wasm(exec_env, func, argc, argv)) {
        printf("call wasm function %s failed. %s\n",func_name,
               wasm_runtime_get_exception(module_inst));
        wasm_runtime_dump_mem_consumption(exec_env);
        assert(0);
    }

    return 0;
}

static wasm_exec_env_t create_exec_env(wasm_module_inst_t* module_inst) {

    wasm_exec_env_t exec_env = wasm_runtime_create_exec_env(*module_inst, 1024 * 1024 * 100);
    if (!exec_env) {
        printf("Create wasm execution environment failed.\n");
        return NULL;
    }

    return exec_env;
}

static wasm_module_inst_t exec_module_instance(wasm_module_t* module) {

    char error_buf[128];

    wasm_module_inst_t module_inst = wasm_runtime_instantiate(*module, STACK_SIZE, HEAP_SIZE,
                                           error_buf, sizeof(error_buf));

    if (!module_inst) {
        printf("Instantiate wasm module failed. error: %s\n", error_buf);
        return NULL;
    }

    return module_inst;
}

static wasm_module_t load_module(char* wasm_path){

    const char *buffer;
    char error_buf[128];
    uint32_t buf_size;
    char wasi_dir_buf[] = ".";
    const char *wasi_dir_list[] = { wasi_dir_buf };

    buffer = bh_read_file_to_buffer(wasm_path, &buf_size);
    if (!buffer) {
        printf("Open wasm app file [%s] failed.\n", wasm_path);
        return NULL;
    }

    wasm_module_t module = wasm_runtime_load(buffer, buf_size, error_buf, sizeof(error_buf));
    if (!module) {
        printf("Load wasm module failed. error: %s\n", error_buf);
        return NULL;
    }

    wasm_runtime_set_wasi_args(module,
                                wasi_dir_list, 1,
                                NULL, 0,
                                NULL, 0,
                                NULL, 0);

    return module;

}

static int start_riscv(char* riscv_image_path) {

#define REG(address) *(volatile unsigned int*)(address)

   //DMEM
    int uio0_fd = open("/dev/uio0", O_RDWR | O_SYNC);
    unsigned char* UDMEM_BASE = (unsigned char*) mmap(NULL, 0x20000, PROT_READ|PROT_WRITE, MAP_SHARED, uio0_fd, 0);

    //IMEM
    int uio2_fd = open("/dev/uio2", O_RDWR | O_SYNC);
    unsigned int* IMEM_BASE = (unsigned int*) mmap(NULL, 0x40000, PROT_READ|PROT_WRITE, MAP_SHARED, uio2_fd, 0);

    //IMEM1
    int uio3_fd = open("/dev/uio3", O_RDWR | O_SYNC);
    unsigned int* IMEM1_BASE = (unsigned int*) mmap(NULL, 0x40000, PROT_READ|PROT_WRITE, MAP_SHARED, uio3_fd, 0);

    //GPIO
    int uio4_fd = open("/dev/uio4", O_RDWR | O_SYNC);
    unsigned int* GPIO_DATA = (unsigned int*) mmap(NULL, 0x4000, PROT_READ|PROT_WRITE, MAP_SHARED, uio4_fd, 0);
    unsigned int* GPIO_TRI = GPIO_DATA + 1;

    FILE* fd = fopen("main", "rb");
    if(fd == NULL) {
            printf("main file not opened\n");
            return NULL;
    }
    if (fseek(fd, 0x1000, SEEK_END) != 0) {
        return NULL;
    }
    int file_size = ftell(fd);
    if (file_size == -1) {
        return NULL;
    }

    FILE* fd1 = fopen("main_b", "rb");
    if(fd1 == NULL) {
            printf("main file not opened\n");
            return NULL;
    }
    if (fseek(fd1, 0x1000, SEEK_END) != 0) {
        return NULL;
    }
    int file_size1 = ftell(fd1);
    if (file_size1 == -1) {
        return NULL;
    }

    printf("RiscV program file size is %d\n", file_size);
    fseek(fd, 0x1000, SEEK_SET);
    fseek(fd1, 0x1000, SEEK_SET);

    int rc = fread(IMEM_BASE, sizeof(unsigned int), 0x40000 / sizeof(unsigned int), fd);
    if (rc < 0) {
            printf("Main File Read error\n");
            return NULL;
    }
    rc = fread(IMEM1_BASE, sizeof(unsigned int), 0x40000 / sizeof(unsigned int), fd1);
    if (rc < 0) {
            printf("Main File Read error\n");
            return NULL;
    }
    sleep(1);
      REG(GPIO_DATA) = 0x04; // LED1
    sleep(1);
      REG(GPIO_DATA) = 0x01; // Reset off

    fclose(fd);
    fclose(fd1);
    return 0;
}

static int start_runtime(void) {

	static char global_heap_buf[1024 * 1024 * 1024];
    int opt;

    char *native_buffer = NULL;
    uint32_t wasm_buffer = 0;
    uint32_t ret;

    RuntimeInitArgs init_args;
    memset(&init_args, 0, sizeof(RuntimeInitArgs));

    // Define an array of NativeSymbol for the APIs to be exported.
    // Note: the array must be static defined since runtime
    //            will keep it after registration
    // For the function signature specifications, goto the link:
    // https://github.com/bytecodealliance/wasm-micro-runtime/blob/main/doc/export_native_api.md

    static NativeSymbol native_symbols[] = {

        {
            "x_gemm", // the name of WASM function name
            x_gemm,   // the native function pointer
            "(iiiiif*i*if*i)",  // the function prototype signature, avoid to use i32
            NULL        // attachment is NULL
        },
        {
            "callback_predict_result", // the name of WASM function name
            callback_predict_result,   // the native function pointer
            "(*iiii)",  // the function prototype signature, avoid to use i32
            NULL        // attachment is NULL
        },
/*
        {
            "get_pow", // the name of WASM function name
            get_pow,   // the native function pointer
            "(ii)i",   // the function prototype signature, avoid to use i32
            NULL       // attachment is NULL
        },
        { "calculate_native", calculate_native, "(iii)i", NULL }
*/
    };

/*
    init_args.mem_alloc_type = Alloc_With_Pool;
    init_args.mem_alloc_option.pool.heap_buf = global_heap_buf;
    init_args.mem_alloc_option.pool.heap_size = sizeof(global_heap_buf);
*/
    init_args.mem_alloc_type = Alloc_With_Allocator;
    init_args.mem_alloc_option.allocator.malloc_func = malloc;
    init_args.mem_alloc_option.allocator.realloc_func = realloc;
    init_args.mem_alloc_option.allocator.free_func = free;

    // Native symbols need below registration phase
    init_args.n_native_symbols = sizeof(native_symbols) / sizeof(NativeSymbol);
    init_args.native_module_name = "env";
    init_args.native_symbols = native_symbols;

    /*if (!wasm_runtime_full_init(&init_args)) {
        printf("Init runtime environment failed.\n");
        return -1;
    }*/
    ret = wasm_runtime_init();
    assert(ret);
    ret = wasm_runtime_register_natives(
  		"env", native_symbols,
  		sizeof(native_symbols) / sizeof(native_symbols[0]));
  	assert(ret);

    return 0;
}

int main(int argc, char** argv) {

    wasm_function_inst_t func = NULL;
    uint32_t ret;
    char filename[256];

    ret = start_runtime();
    assert(ret == 0);
#if WASI_GEMM_RISC_V
    ret = start_riscv(".");
#endif
    //wasm_module_t module = load_module("test.wasm");
    wasm_module_t module = load_module("tracking.wasm");
    assert(module);

    wasm_module_inst_t module_inst = exec_module_instance(&module);
    assert(module_inst);

    g_exec_env = create_exec_env(&module_inst);
    assert(g_exec_env);

    assert(wasm_runtime_is_wasi_mode(module_inst));

    call_wasm_function(g_exec_env, "_start", 0, argv);

/*
    char *datacfg = argv[1];
    char *cfgfile = argv[2];
    char *weightfile = argv[3];
*/

	ret = yolo_initialize("coco.data", "signate.cfg", "signate_final.weights");
    assert(ret == 0);

    char *fname = argv[1];

    FILE* fp = fopen(fname, "r");
	if(fp == NULL) {
		printf("%s file not open!\n", fname);
		return -1;
	}    
    while(fgets(filename, 256, fp) != NULL) {

        char* p = strchr( filename, '\n' );
        /* 改行文字があった場合 */
        if ( p != NULL )
        {
            /* 改行文字を終端文字に置き換える */
            *p = '\0';
        }

        char output_filename[256];
        snprintf(output_filename, 256, "%s_output", filename);
        printf("%s \n", filename);
        printf("%s \n", output_filename);
	    ret = test_detector(filename, output_filename);
        assert(ret == 0);
    }
    fclose(fp);

    wasm_runtime_destroy_exec_env(g_exec_env);
    g_exec_env = NULL;

    wasm_runtime_deinstantiate(module_inst);

    wasm_runtime_unload(module);

    print_save_json_result("test_00.mp4");

	return 0;
}
