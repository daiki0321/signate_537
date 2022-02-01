#include "darknet.h"

static char **names;
static image **alphabet;

void callback_predict_result(detection *dets, int total, int classes, int w, int h);

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

typedef struct detection_orig {
    int class;
    int left;
    int top;
    int right;
    int bottom;
    float score;
}detection_orig;

static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

detection_orig* convert_result_to_native(detection *dets, int total, float thresh, int classes, int w, int h)
{
    int i, j;

    if (total < 1) {
        return NULL;
    }

    detection_orig* buf = (detection_orig*)malloc(sizeof(detection_orig) * total);

    for(i = 0; i < total; ++i){

        box b = dets[i].bbox;

        int left  = (b.x-b.w/2.)*w;
        int right = (b.x+b.w/2.)*w;
        int top   = (b.y-b.h/2.)*h;
        int bot   = (b.y+b.h/2.)*h;

        if(left < 0) left = 0;
        if(right > w-1) right = w-1;
        if(top < 0) top = 0;
        if(bot > h-1) bot = h-1;

        buf[i].left = left;
        buf[i].top = top;
        buf[i].right = right;
        buf[i].bottom = bot;

        int class = -1;
        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j] > thresh){
                if ((class < 0) || (dets[i].prob[j] > buf[i].score )) {
                    class = j;
                    buf[i].score = dets[i].prob[j];
                    buf[i].class = class;
                }
            }
        }
        if (class == -1) {
            memset(&buf[i], 0, sizeof(detection_orig));
        }

        fprintf(stderr, "class = %d left = %d top = %d right = %d bottom = %d score =%f\n",
        buf[i].class, buf[i].left, buf[i].top, buf[i].right, buf[i].bottom, buf[i].score);
    }
    return buf;
}

detection_orig* test_detector(network* net, char *filename, float thresh, float hier_thresh, char *outfile, int* pnboxes, int fullscreen)
{
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    detection_orig* det_native;

    if(filename){
        strncpy(input, filename, 256);
    } else {
        printf("Enter Image Path: ");
        fflush(stdout);
        input = fgets(input, 256, stdin);
        if(!input) return NULL;
        strtok(input, "\n");
    }
    image im = load_image_color(input,0,0);
    image sized = letterbox_image(im, net->w, net->h);
    //image sized = resize_image(im, net->w, net->h);
    //image sized2 = resize_max(im, net->w);
    //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
    //resize_network(net, sized.w, sized.h);
    layer l = net->layers[net->n-1];

    float *X = sized.data;
    time=what_time_is_it_now();
    network_predict(net, X);
    int nboxes;
    fprintf(stderr, "%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
    fprintf(stderr, "nboxes = %d\n", nboxes);
    //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
    
    det_native = convert_result_to_native(dets, nboxes, thresh, l.classes, im.w, im.h);
    *pnboxes = nboxes;

    free_detections(dets, nboxes);
    if(outfile){
        save_image(im, outfile);
    }
    else{
        save_image(im, "predictions");
#ifdef OPENCV
        make_window("predictions", 512, 512, 0);
        show_image(im, "predictions", 0);
#endif
    }

    free_image(im);
    free_image(sized);

    return det_native;

}

network* yolo_initialize(char *datacfg, char *cfgfile, char *weightfile) {

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "coco.names");
    names = get_labels(name_list);

    alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    return net;
}

int main(int argc, char** argv) {

    printf("This is main function\n");

    return 0;

}