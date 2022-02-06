#include "tracker.h"
#include "model.h"
#include "FeatureTensor.h"
#include <algorithm>

typedef struct detection_orig {
    int class_num;
    int left;
    int top;
    int right;
    int bottom;
    float score;
}detection_orig;

tracker mytracker(0.8, 100, 0.6, 50, 2, 1.0);
std::shared_ptr<FeatureTensor> featureTensor(new FeatureTensor("tracker/RUNNINGDATA/tensor_networks/111.meta", "tracker/RUNNINGDATA/tensor_networks/mars-small128.ckpt-68577"));
static int frameID = 0;

#ifdef __cplusplus
extern "C" {
#endif

void tracker_init(void) {
        //mytracker(0.5, 10, 0.5, 3, 0, 0.5);
        return;
}

bool tracker_update(cv::Mat frame, int nboxes, detection_orig* detection) {

    DETECTIONS detections;

    if(nboxes == 0) {
        return false;
    }

    DETECTION_ROW detection_row;
    for(int i = 0; i < nboxes; i++) {
        detection_row.tlwh = DETECTBOX(detection[i].left, detection[i].top, detection[i].right - detection[i].left, detection[i].bottom - detection[i].top);
        detection_row.class_num = detection[i].class_num;
        detection_row.confidence = detection[i].score;

        if (detection_row.confidence > .5) {
            printf("top = %d left = %d width = %d height = %d\n", 
                detection[i].top, detection[i].left, detection[i].right - detection[i].left, detection[i].bottom - detection[i].top);
            cv::Rect rect = cv::Rect((int)detection[i].left, detection[i].top, detection[i].right - detection[i].left, detection[i].bottom - detection[i].top);
            cv::rectangle(frame, rect, (0, 0, 0), 2, cv::LINE_8);
            detections.push_back(detection_row);
        }
    }

    //TENSORFLOW get rect's feature.
    if(featureTensor->getRectsFeature(frame, detections) == false) 
    {
        printf("Tensorflow get feature failed!");
        return false;
    }

    mytracker.predict();
    mytracker.update(detections);
    std::vector<RESULT_DATA> result;

    for(Track& track : mytracker.tracks) 
    {
        printf("%s %d %d %d \n", __func__, __LINE__, track.is_confirmed(), track.time_since_update);
        if(!track.is_confirmed() || track.time_since_update > 1) continue;
        result.push_back(std::make_pair(std::make_pair(track.track_id, track.detection_class), std::make_pair(track.to_tlwh(), track.color)));
    }
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
    for(unsigned int k = 0; k < result.size(); k++) 
    {
        DETECTBOX tmp = result[k].second.first;
        std::string det_class = result[k].first.second;
        cv::Scalar color = result[k].second.second;
        fprintf(stderr, "ID = %d class = %s X1 = %f Y1 = %f X2 = %f Y2 = %f \n", 
            result[k].first.first, result[k].first.second.c_str(), tmp(0), tmp(1), tmp(0) + tmp(2), tmp(1) + tmp(3));
        cv::Rect rect = cv::Rect((int)tmp(0), (int)tmp(1), (int)tmp(2), (int)tmp(3));
        cv::rectangle(frame, rect, color, 2, cv::LINE_8);
        ss << result[k].first.first << " - " << det_class;
        cv::putText(frame, ss.str(), cv::Point(rect.x, rect.y), CV_FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        ss.str("");
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

#ifdef __cplusplus
};
#endif

