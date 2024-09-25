import paddlehub as hub
import insightface_paddle as face
import logging
logging.basicConfig(level=logging.INFO)
import cv2
import json
import numpy as np
import time
import datetime
from src.VideoStream import VideoStream
from PIL import Image



face_detector = hub.Module(name="pyramidbox_lite_mobile",ignore_env_mismatch = True)
parser = face.parser()
args = parser.parse_args()
recognizer = []





def detect_face(image):
    global shrink_face_det, det_threshold
    result = face_detector.face_detection([image],
                   use_gpu=args.use_gpu,
                   shrink=shrink_face_det,
                   confs_threshold=det_threshold)
    #print(len(result))
    box_list = result[0]['data']
    return box_list
    
    
def preprocess_img(img, box_list=None):
        if box_list==[]:return None
        if box_list is None:
            height, width = img.shape[:2]
            box_list = [np.array([0, 0, 0, 0, width, height])]
        batch = []
        cnt = 0
        for idx, box in enumerate(box_list):
            xmin, ymin, xmax, ymax = int(box['left']), int(box['top']), int(box['right']), int(box['bottom'])
            face_img = img[ymin:ymax, xmin:xmax, :]
            face_img = cv2.resize(face_img, (112, 112))
            batch.append(face_img)
            cnt += 1
        #print(batch[0].shape)
        return batch
    
def recognize_face(image):
    if image is None:
        return 
    img = image[:, :, ::-1]
    res = list(recognizer.predict(img))
    #features = list(recognizer.rec_predictor.predict(img,box_list))
    #label=[]
    #label = recognizer.rec_predictor.retrieval(features)
    labels = res[0]['labels']
    return labels

def draw_boundary_boxes(image, box_list, labels):

    if box_list is None: return
    for box,label in zip(box_list,labels):
        score = "{:.2f}".format(box['confidence'])
        if float(score)<bbox_threshold:continue
        x_min, y_min, x_max, y_max = int(box['left']), int(box['top']), int(box['right']), int(box['bottom'])

        # Draw the bounding box
        if(label[0]=="unknown"):
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            label_f = label[0]+score
            cv2.putText(image, label_f, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label_f = label[0]+score
            cv2.putText(image, label_f, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Put the label text near the box
        

def detection_video_stream(camera_urls):
    # Create VideoStream instances for each camera
    cameras = [VideoStream(url) for url in camera_urls]

    # Start reading frames from each camera
    for camera in cameras:
        camera.start()
    
    # Display frames from each camera in separate windows
    while cameras:
        Fps_time = time.time()
        canvas = np.zeros((1280, 1920, 3), dtype=np.uint8)
        cv2.namedWindow('Canvas Window', cv2.WINDOW_FULLSCREEN)

        # 将窗口移动到指定位置 (x, y)
        cv2.moveWindow('Canvas Window', 100, 00)
        for i, camera in enumerate(cameras):
            frame = camera.read()
            if(frame is None):
                continue
                #print(frame.shape)
                #h=frame.shape[0]
                #w=frame.shape[1]
            #print([h,w])
            crop_frame = frame[:,100:580]
            resized_frame = cv2.resize(crop_frame, (640,480), interpolation= cv2.INTER_LINEAR)
            if resized_frame is not None:
               detst_time = time.time()
               box_list = detect_face(resized_frame)
               detect_time = time.time()-detst_time
               #print(f'detect time:{detect_time}')
               if box_list !=[]: 
                   #print(box_list)
                   faces = preprocess_img(resized_frame, box_list)
                   recst_time = time.time()
                   labels = [recognize_face(face) for face in faces]
                   #print(labels)
                   rec_time=time.time()-detect_time-recst_time
                   #print(f'REC time:{rec_time}')
                   draw_boundary_boxes(resized_frame,box_list,labels)
                   #cv2.imshow(f"Camera_f {i+1}", faces[0])
               if(i<3):
                   x = int(i*640)
                   canvas[0:480, x:x+640] = resized_frame
               else:
                   x = int((i-3)*640)
                   canvas[480:960, x:x+640] = resized_frame
               cv2.imshow('Canvas Window', canvas)
               
               
		#resize frame to match the input size of the model
		
		
            else:
                print(f"Camera {cameras.index(camera)+1} disconnected")
                camera.stop()
                cv2.destroyAllWindows()
                cameras.remove(camera)
        oneframe = time.time()-Fps_time
        #print(f'FPS:{1/oneframe}') 

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    # Stop reading frames from each camera
    for camera in cameras:
        camera.stop()

    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    
    with open('camera_urls.json', 'r') as f:
        camera_urls = json.load(f)
    with open('recognition_condition.json', 'r') as f:
        config = json.load(f)




    args.use_gpu=config['use_gpu']
   
    args.enable_mkldnn = True
    args.det = False
    args.rec = True
    args.rec_thresh = config['rec_thresh']
    args.cdd_num = config['cdd_num']
    args.index = config['index']
    args.rec_model = "Models/mobileface_v1.0_infer"
    recognizer = face.InsightFace(args)
    det_threshold = config['det_threshold']
    bbox_threshold = config['bbox_threshold']
    shrink_face_det = config['shrink_face_det']
    
    detection_video_stream(camera_urls)



