#include <assert.h>
#include <stdio.h>

#include "utils.h"
#include "wasm_export.h"
#include "darknet.h"

extern wasm_exec_env_t g_exec_env;

/* this will store wasm local app addr */
static network* g_net;

typedef struct detection_orig {
    int class;
    int left;
    int top;
    int right;
    int bottom;
    float score;
}detection_orig;

void callback_predict_result(wasm_exec_env_t exec_env, void* result, int total, int classes, int w, int h)
{
    int i, j;

    wasm_function_inst_t module_inst = get_module_inst(exec_env);
    assert(module_inst);

    printf("callback_predict_result %x %d %d %d %d \n", result, total, classes, w, h);

    detection* dets = (detection*)result;

    for(i = 0; i < 32; i++) {
        fprintf(stderr, "0x%x\n", *((uint32_t*)result+i));
    }

    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        /* Todo: dets[i].prob is wasm module addr,can't refer to this value */
        /* And structure size is different between native and wasm, difficult to parse */
        /* for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(stderr, "%d %f %f %f %f %f\n", j, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        } */
        fprintf(stderr, "%f %f %f %f\n", xmin, ymin, xmax, ymax);
    }
    return;
}

int test_detector(char *filename, char *outfile)
{
    uint32_t argv[7] = {0};
    char *native_buffer = NULL;
    int* nboxes;
    detection_orig* buf;

    wasm_function_inst_t module_inst = get_module_inst(g_exec_env);
    assert(module_inst);

    argv[0] = g_net;

    uint32_t wasm_buffer_1 =
        wasm_runtime_module_malloc(module_inst, strlen(filename) + 1, (void **)&native_buffer);
    assert(wasm_buffer_1);
    assert(native_buffer);
    strncpy(native_buffer, filename, strlen(filename) + 1);
    argv[1] = wasm_buffer_1;

    *(float *)(argv + 2) = .5;
    *(float *)(argv + 3) = .5;

    uint32_t wasm_buffer_2 =
        wasm_runtime_module_malloc(module_inst, strlen(outfile) + 1, (void **)&native_buffer);
    assert(wasm_buffer_2);
    assert(native_buffer);
    strncpy(native_buffer, outfile, strlen(outfile) + 1);
    argv[4] = wasm_buffer_2;

    uint32_t wasm_buffer_3 =
        wasm_runtime_module_malloc(module_inst, sizeof(int), (void **)&native_buffer);
    assert(wasm_buffer_3);
    assert(native_buffer);
    argv[5] = wasm_buffer_3;
    nboxes = native_buffer;
    *nboxes = 0;

    argv[6] = 0;

    uint32_t ret = call_wasm_function(g_exec_env, "test_detector", 7, argv);
    assert(ret == 0);

    printf("[native] nboxes = %d \n", *nboxes);

    bool ok = wasm_runtime_validate_app_addr(module_inst, argv[0], sizeof(detection_orig)*(*nboxes));
    assert(ok);

    if (*nboxes > 0) {
        buf = malloc(sizeof(detection_orig)*(*nboxes));
        memcpy(buf, wasm_runtime_addr_app_to_native(module_inst, argv[0]), sizeof(detection_orig)*(*nboxes));
        for(int i = 0; i < *nboxes; i++) {
            printf("class = %d left = %d top = %d right = %d bottom = %d score =%f\n",
        buf[i].class, buf[i].left, buf[i].top, buf[i].right, buf[i].bottom, buf[i].score);
        }
        wasm_runtime_module_free(module_inst, argv[0]);
    }

    wasm_runtime_module_free(module_inst, wasm_buffer_1);
    wasm_runtime_module_free(module_inst, wasm_buffer_2);
    wasm_runtime_module_free(module_inst, wasm_buffer_3);

    return 0;

}

int yolo_initialize(char *datacfg, char *cfgfile, char *weightfile)
{

    char *native_buffer = NULL;
    uint32_t ret;
    bool ok;
    uint32_t argv[3] = {0};

/*
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
*/
    wasm_function_inst_t module_inst = get_module_inst(g_exec_env);
    assert(module_inst);

    uint32_t wasm_buffer_1 =
        wasm_runtime_module_malloc(module_inst, strlen(datacfg) + 1, (void **)&native_buffer);
    assert(wasm_buffer_1);
    assert(native_buffer);
    strncpy(native_buffer, datacfg, strlen(datacfg) + 1);
    argv[0] = wasm_buffer_1;

    uint32_t wasm_buffer_2 =
        wasm_runtime_module_malloc(module_inst, strlen(cfgfile) + 1, (void **)&native_buffer);
    assert(wasm_buffer_2);
    assert(native_buffer);
    strncpy(native_buffer, cfgfile, strlen(cfgfile) + 1);
    argv[1] = wasm_buffer_2;


    uint32_t wasm_buffer_3 =
        wasm_runtime_module_malloc(module_inst, strlen(weightfile) + 1, (void **)&native_buffer);
    assert(wasm_buffer_3);
    assert(native_buffer);
    strncpy(native_buffer, weightfile, strlen(weightfile) + 1);
    argv[2] = wasm_buffer_3;

    ret = call_wasm_function(g_exec_env, "yolo_initialize", 3, argv);
    assert(ret == 0);

    ok = wasm_runtime_validate_app_addr(module_inst, argv[0], sizeof(network));
    assert(ok);

    g_net = argv[0];

    wasm_runtime_module_free(module_inst, wasm_buffer_1);
    wasm_runtime_module_free(module_inst, wasm_buffer_2);
    wasm_runtime_module_free(module_inst, wasm_buffer_3);

    return 0;

}