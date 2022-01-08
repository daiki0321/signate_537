#import libraries
import os
import json

#define paths
path_labels    = '/home/guest/work/akasaka/signate_537/train_annotations'      
path_images    = '/home/guest/work/akasaka/yolov5/datasets/signate/train_imgs_640_640/images' #path where, the are the decoded train images
#path_out_file  = 'data_for_yolo_training.txt'
train_text = 'train.txt'
valid_text = 'valid.txt'

#array that defines which classes have to be extracted. 
classes = ['Car', 'Pedestrian', 'Truck', 'Signal', 'Signs', 'Bicycle', 'Motorbike', 'Bus', 'Svehicle', 'Train']

train_list_file    = open(os.path.join(path_images, train_text) , "w")
valid_list_file    = open(os.path.join(path_images, valid_text) , "w")

#out_file    = open(path_out_file, "w")
annotations = os.listdir(path_labels)

for i in range (0, len(annotations)):#here we browse all videos
    video_name = annotations[i].split('/')[-1].split('\\')[-1].split('.')[0]

    try: os.makedirs(os.path.join(path_images, "labels", video_name),  exist_ok=True)
    except OSError:
        print ('cannot create directory '+ os.path.join(path_images, video_name))
        exit()
    
    data       = json.load(open(os.path.join(path_labels, annotations[i])))
    
    for v in range (0,600): #here we browse all frames. Single movie has 600 frames
        img_name     = path_images+'/'+video_name+'/'+str(v)+".jpg"
        if v % 100 :
            train_list_file.write(img_name+'\n')
        else :
            valid_list_file.write(img_name+'\n')

        path_out_file = path_images+'/labels/'+video_name+'/'+str(v)+".txt"
        out_file    = open(path_out_file, "w")
        
        labels       = data['sequence'][v]
        str_to_write = ""
        for c in range (0, len(classes)):
            try:
                for inst in data['sequence'][v][classes[c]]:
                    box           = inst['box2d']
                    str_to_write += str(c)+' '+str('{:.06f}'.format((box[0] + box[2])/2/1936))+' '+str('{:.06f}'.format((box[1] + box[3])/2/1216))+' '+str('{:.06f}'.format((box[2] - box[0])/1936))+' '+str('{:.06f}'.format((box[3] - box[1])/1216))+'\n'
            except:
                continue #nothing, the class is just not presented in the frame
         
        try :
            out_file.write(str_to_write)
            out_file.close()
        except:
            continue #nothing, the class is just not presented in the frame
        
train_list_file.close()
valid_list_file.close()

