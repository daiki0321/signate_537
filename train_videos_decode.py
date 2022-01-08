#import libraries
import cv2
import os

#define paths
path_to_videos    = '/home/guest/work/akasaka/signate_537/train_videos'
path_to_save_imgs = '/home/guest/work/akasaka/yolov5/datasets/signate/train_imgs_640_640/images'

#browse all videos, decode them, and save into folders
list_of_videos =[os.path.join(root, name) for root, dirs, files in os.walk(path_to_videos) for name in files]
for v in range (0, len(list_of_videos)):
    video_name = list_of_videos[v].split('/')[-1].split('\\')[-1].split('.')[0]
    print(video_name)
    
    try: os.makedirs(os.path.join(path_to_save_imgs, video_name),  exist_ok=True)
    except OSError:
        print ('cannot create directory '+ os.path.join(path_to_save_imgs, video_name))
        exit()

    stream = cv2.VideoCapture(list_of_videos[v])
    #stream.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
    #stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
    for i in range (0, 10000):
        (grabbed, frame) = stream.read()
        if not grabbed: break
        frame = cv2.resize(frame, (640,640), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(path_to_save_imgs, video_name, str(i)+'.jpg'), frame)
    
print('done')
