import paddlehub as hub
import insightface_paddle as face
import cv2
import os
import time
import json
import sqlite3

# camera_url = "rtsp://admin:unitedvigi@10.10.2.5/stream2"

# parser = face.parser()
# args = parser.parse_args()
# args.use_gpu=True
# args.build_index = "./Dataset/index.bin"
# args.img_dir = "./Dataset/gallery"
# args.label = "./Dataset/gallery/labels.txt"
# det_threshold = 0.9

face_detector = hub.Module(name="pyramidbox_lite_mobile")
predictor = []



def draw_bounding_boxes(image, faces):
    # Draw bounding boxes on the image
    for face in faces:
        left = int(face['left'])
        right = int(face['right'])
        top = int(face['top'])
        bottom = int(face['bottom'])
        confidence = face['confidence']

        # Draw a rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 1)
        label = f": {confidence:.2f}"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
    return image
  
def crop_and_save_face(img, filepath, box_list):
  
  for box in box_list:
    xmin = int(box['left'])
    ymin = int(box['top'])
    xmax = int(box['right'])
    ymax = int(box['bottom'])
    face_img = img[ymin:ymax, xmin:xmax, :]
    cv2.imwrite(filepath, face_img)
    
def write_to_file(filepath, person_name, filename="./app/static/images/labels.txt"):
    modified_filepath = "./" + os.path.join(person_name, os.path.basename(filepath))
    with open(filename, 'a') as f:
        f.write("{}\t{}\n".format(modified_filepath, person_name))
        
def enter_person_name():
    # Create a directory for the person's name
    conn = sqlite3.connect('./instance/site.db')

    # 建立游標
    cursor = conn.cursor()

    # 執行 SQL 查詢
    cursor.execute("SELECT * FROM enrollment ORDER BY id DESC LIMIT 1")

    # 獲取所有結果
    row = cursor.fetchone()
    person_name = row[1]
    print(person_name)
    output_dir = os.path.join("./app/static/images", person_name)
    os.makedirs(output_dir, exist_ok=True)
    return person_name, output_dir

def capture_face_images(camera_url, person_name, output_dir):
    # Initialize the count
    cnt = 0

    # Start the webcam
    cap = cv2.VideoCapture(camera_url)

    end_time = time.time() + 7
    frame_count = 0

    while time.time() < end_time:
        
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        frame_count += 1

        # Perform face detection on the frameß
        crop_frame = frame[:,100:580]
        resized_frame = cv2.resize(crop_frame, (640,480), interpolation= cv2.INTER_LINEAR)
        result = face_detector.face_detection([resized_frame],
                   use_gpu=args.use_gpu,
                   shrink=0.4,
                   confs_threshold=det_threshold)
        box_list = result[0]['data']
        img = draw_bounding_boxes(resized_frame, box_list)
        #cv2.imshow(f"Camera", img)
        # Crop and save the detected faces
        if frame_count % 2 == 0 and box_list:
            filename = '{}_{}.jpeg'.format(person_name, cnt)
            filepath = os.path.join(output_dir, filename)
            crop_and_save_face(resized_frame, filepath, box_list)
            write_to_file(filepath, person_name)
            cnt += 1
            
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def video_show():
    cap = cv2.VideoCapture(camera_url1)

    end_time = time.time() + 7

    while time.time() < end_time:
        ret, frame = cap.read()
        crop_frame = frame[:,100:580]
        resized_frame = cv2.resize(crop_frame, (640,480), interpolation= cv2.INTER_LINEAR)
        #cv2.imshow(f"Camera", frame)
        #cv2.imshow(f"Crop",resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    with open('enroll_condition.json', 'r') as f:
        config = json.load(f)

    camera_url1 = config['camera_url1']
    camera_url2 = config['camera_url2']
    parser = face.parser()
    args = parser.parse_args()
    args.use_gpu=config['use_gpu']
    args.build_index = config['build_index']
    args.img_dir = config['img_dir']
    args.label = config['label']
    det_threshold = config['det_threshold']

    predictor =  face.InsightFace(args)

    person_name, output_dir = enter_person_name()
    capture_face_images(camera_url1, person_name, output_dir)
    time.sleep(5)
    capture_face_images(camera_url2, person_name, output_dir)
    predictor.build_index()
