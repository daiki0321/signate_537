//#include "tracker.h"
#include "deepsort.h"
#include "model.h"
#include <algorithm>

#include "json.hpp"
// for convenience
using json = nlohmann::json;
json json_result;

typedef struct detection_orig {
    int class_num;
    int left;
    int top;
    int right;
    int bottom;
    float score;
}detection_orig;

Deep_sort Tracker;
static int frameID = 0;

static void init_json(int frame_cnt){

    json_result[frame_cnt] = json::object(); 
}

const std::vector<std::string> class_name = {"Car", "Pedestrian"};
const std::vector<cv::Scalar> color_map = {cv::Scalar(0,0,255),cv::Scalar(255, 0, 0),cv::Scalar(0, 255, 0)
                                ,cv::Scalar(255,255,0),cv::Scalar(255,0,255),cv::Scalar(0,255,255)
                                ,cv::Scalar(180,54,0),cv::Scalar(54,200,54),cv::Scalar(255,211,155)};

static void update_json_result(int frame_cnt, int id, std::string det_class, int left, int top, int right, int bottom){

        json bbox = {{"box2d", {left, top, right, bottom}}, {"id", id}};
        //json current_frame = result[frame_cnt][det_class];

        json_result[frame_cnt][det_class].push_back(bbox);

        //std::cout << current_frame.dump(4) << std::endl;

}

#ifdef __cplusplus
extern "C" {
#endif

void tracker_init(void) {
        //Deep_sort(float max_cosine_distance, int nn_budget, float max_iou_distance, int max_age, int n_init)
        Tracker = Deep_sort(0.2, 100, 0.7, 30, 3);
        return;
}

bool tracker_update(cv::Mat frame, int nboxes, detection_orig* detection) {

    DETECTIONS detections;
    std::deque<cv::Point> line_point;

    if(nboxes == 0) {
        return false;
    }

    DS_DetectObjects detect_objects;
    
    for (int i=0;i<nboxes;i++){
        DS_Rect rec;DS_DetectObject obj;
        int x1 = (int)(detection[i].left);
        int y1 = (int)(detection[i].top);
        int w = (int)(detection[i].right - detection[i].left);
        int h = (int)(detection[i].bottom - detection[i].top);
        // int x1 = (int)((dets[i].bbox.x - dets[i].bbox.w/2)*width);
        // int y1 = (int)((dets[i].bbox.y - dets[i].bbox.h/2)*height);
        // int w = (int)(dets[i].bbox.w*width);
        // int h = (int)(dets[i].bbox.h*height);
        //int x1 = (int)(x-w/2.0);
        //int y1 = (int)(y-h/2.0);
        if(x1<0) x1=0;
        if(y1<0) y1=0;
        if((x1+w)>frame.size[1]) w=frame.size[1]-x1;
        if((y1+h)>frame.size[0]) h=frame.size[0]-y1;
        float prob = detection[i].score;
        rec.x = x1;rec.y = y1;rec.width = w;rec.height = h;
        obj.class_id = detection[i].class_num;obj.rect = rec;obj.confidence = prob;
        if (detection[i].score > .5) {
            cv::Rect rect = cv::Rect((int)detection[i].left, detection[i].top, detection[i].right - detection[i].left, detection[i].bottom - detection[i].top);
            cv::rectangle(frame, rect, (0, 0, 0), 2, cv::LINE_8);
            detect_objects.push_back(obj);
        }
    }

    Tracker.update(detect_objects,line_point,frame);

    DS_TrackObjects track_objects = Tracker.get_detect_obj();

    //TENSORFLOW get rect's feature.
    /*if(featureTensor->getRectsFeature(frame, detections) == false) 
    {
        printf("Tensorflow get feature failed!");
        return false;
    }*/

    //mytracker.predict();
    //mytracker.update(detections);
    //std::vector<RESULT_DATA> result;

    /*for(Track& track : mytracker.tracks) 
    {
        printf("%s %d %d %d \n", __func__, __LINE__, track.is_confirmed(), track.time_since_update);
        if(!track.is_confirmed() || track.time_since_update > 1) continue;
        result.push_back(std::make_pair(std::make_pair(track.track_id, track.detection_class), std::make_pair(track.to_tlwh(), track.color)));
    }*/
/*
    std::stringstream ss;
    for(unsigned int k = 0; k < result.size(); k++) 
    {
        DETECTBOX tmp = result[k].second.first;
        std::string det_class = result[k].first.second;
        cv::Scalar color = result[k].second.second;
        cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
        cv::rectangle(frame, rect, color, 2, cv::LINE_8);
        ss << result[k].first.first << " - " << det_class;
        cv::putText(frame, ss.str(), cv::Point(rect.x, rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        ss.str("");
    }
    imshow("DeepSortTracking", frame);
*/

    std::stringstream ss;
    init_json(frameID);
    for(auto oloop : track_objects) 
    {

        if(oloop.rect.x<0)
        {
            oloop.rect.x = 0;
        }
        if(oloop.rect.y<0)
        {
            oloop.rect.y = 0;
        }
        if((oloop.rect.x + oloop.rect.width)>frame.size[1])
        {
            oloop.rect.width = frame.size[1]-oloop.rect.x;
        }
        if((oloop.rect.y + oloop.rect.height)>frame.size[0])
        {
            oloop.rect.height = frame.size[0]-oloop.rect.y;
        }   
        //DETECTBOX tmp = result[k].second.first;
        //std::string det_class = result[k].first.second;
        //cv::Scalar color = result[k].second.second;
        fprintf(stderr, "ID = %d class = %d X1 = %d Y1 = %d X2 = %d Y2 = %d \n", 
            oloop.track_id, oloop.class_id, oloop.rect.x, oloop.rect.y, oloop.rect.x + oloop.rect.width, oloop.rect.y + oloop.rect.height);
        cv::Rect rect = cv::Rect(oloop.rect.x, oloop.rect.y, oloop.rect.width, oloop.rect.height);
        cv::rectangle(frame, rect, color_map[oloop.class_id], 2, cv::LINE_8);
        ss << oloop.track_id << " - " << class_name[oloop.class_id];
        cv::putText(frame, ss.str(), cv::Point(rect.x, rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.8, color_map[oloop.class_id], 2);
        ss.str("");
        update_json_result(frameID, oloop.track_id, class_name[oloop.class_id], oloop.rect.x, oloop.rect.y, oloop.rect.x + oloop.rect.width, oloop.rect.y + oloop.rect.height);
    }

    char output_filename[256];
    snprintf(output_filename, 256, "test_%d.jpg", frameID);
    cv::imwrite(output_filename, frame);
    frameID++;

    return true;
}

void tracker_update_with_filename(char* filename, int nboxes, detection_orig* detection) {

    cv::Mat frame = cv::imread(filename);

    bool ok = tracker_update(frame, nboxes, detection);
    assert(ok);

}

void print_save_json_result(char* filename) {

    json j;

    j[filename] = json_result;

    std::cout << j.dump(4) << std::endl;

    return;
}

#ifdef __cplusplus
};
#endif

