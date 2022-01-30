#include <assert.h>
#include <stdio.h>

#include "wasm_export.h"
#include "utils.h"

#include "darknet.h"

#define STACK_SIZE 8 * 1024
#define HEAP_SIZE 1024 * 1024 * 1024

wasm_exec_env_t g_exec_env = NULL;

static void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

static void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

static void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

static void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

x_gemm_cpu(wasm_exec_env_t exec_env, int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    wasm_function_inst_t module_inst = get_module_inst(g_exec_env);

    {
    printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    }
}

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
            "x_gemm_cpu", // the name of WASM function name
            x_gemm_cpu,   // the native function pointer
            "(iiiiif*i*if*i)",  // the function prototype signature, avoid to use i32
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

	printf("This is main function\n");

    ret = start_runtime();
    assert(ret == 0);

    //wasm_module_t module = load_module("test.wasm");
    wasm_module_t module = load_module("tracking.wasm");
    assert(module);

    wasm_module_inst_t module_inst = exec_module_instance(&module);
    assert(module_inst);

    g_exec_env = create_exec_env(&module_inst);
    assert(g_exec_env);

    assert(wasm_runtime_is_wasi_mode(module_inst));

    call_wasm_function(g_exec_env, "_start", 0, argv);

	//run_yolo(argc, argv);

    wasm_runtime_destroy_exec_env(g_exec_env);
    g_exec_env = NULL;

    wasm_runtime_deinstantiate(module_inst);

    wasm_runtime_unload(module);

	return 0;
}