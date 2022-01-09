import os
import json
import glob

files = sorted(glob.glob("/home/daiki0321/workspace/signate_537/inference_result/*"))
for file in files:
    print(file)

#define paths
path_labels    = '/home/daiki0321/workspace/signate_537/inference_result'      

classes = ['Car', 'Pedestrian', 'Truck', 'Signal', 'Signs', 'Bicycle', 'Motorbike', 'Bus', 'Svehicle', 'Train']

predictions = {}
frame_data = [{}] * 150

last_frame_id = 0

#textFilename = "test_00.txt"

for textFilename in files:
    with open(textFilename, "r") as f_txt:
        videoFilename = os.path.splitext(os.path.basename(textFilename))[0] + ".mp4"

        pedestrian_result_data = []
        car_result_data = []

        while True:
            textdata = f_txt.readline().split()
            if textdata == []:
                break
                
            frameId = int(textdata[0])
            classId = textdata[1]
            track_id = textdata[2]
            bbox = textdata[3:7]

            if (last_frame_id != frameId):

                if car_result_data and pedestrian_result_data:
                    frame_data[last_frame_id - 1] = {"Car": car_result_data, "Pedestrian": pedestrian_result_data}

                elif car_result_data:
                    print(car_result_data)
                    frame_data[last_frame_id - 1] = {"Car": car_result_data}

                elif pedestrian_result_data:
                    print(pedestrian_result_data)
                    frame_data[last_frame_id- 1] = {"Pedestrian": pedestrian_result_data}

                pedestrian_result_data = []
                car_result_data = []
                last_frame_id = frameId

            if ((abs(int(bbox[2]) - int(bbox[0])) * abs(int(bbox[3]) - int(bbox[1]))) >= 1024):

                if classId == "0":
                    car_result_data.append({"id": int(track_id), "box2d": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]) ]})
                elif classId == "1":
                    pedestrian_result_data.append({"id": int(track_id), "box2d": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]) ]})

        if car_result_data and pedestrian_result_data:
            frame_data[last_frame_id - 1] = {"Car": car_result_data, "Pedestrian": pedestrian_result_data}

        elif car_result_data:
            frame_data[last_frame_id - 1] = {"Car": car_result_data}

        elif pedestrian_result_data:
            frame_data[last_frame_id- 1] = {"Pedestrian": pedestrian_result_data}

    predictions[videoFilename] = frame_data.copy()

with open("sample_submit.json","w") as f_json:
    json.dump(predictions, f_json, indent=4)

    
