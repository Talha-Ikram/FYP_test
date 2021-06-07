import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import sys
import copy
import threading
import tensorflow as tf
from tensorflow import keras
from scipy.spatial import distance as dist
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from keras.models import load_model
import easygui

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.01, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

global result_thread, running_thread, hist_thread
result_thread = []
running_thread = False
hist_thread = []

def threader(frame, image, input_size, infer, encoder, nms_max_overlap, tracker, model, thresh):
        global result_thread
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny  == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]
        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        #try to delete low value humans only 
        tester = np.float32(0)
        deleted_scores = []
        for i in range(len(scores)):
          if scores[i]<0.5:
            if classes[i] == tester:
              deleted_scores.append(i)    
        bboxes = np.delete(bboxes, deleted_scores, axis=0)
        scores = np.delete(scores, deleted_scores, axis=0)
        names = np.delete(names, deleted_scores, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # Compute Centroids and Depth
        cord_arr = dict()
        for track in tracker.tracks:
          if not track.is_confirmed() or track.time_since_update > 1 or track.get_class()!='Person':
            continue
          bbox = track.to_tlbr()
          cord_arr[track.track_id] = []
          cord_arr[track.track_id].append((int(bbox[0])+int(bbox[2]))/2)
          cord_arr[track.track_id].append((int(bbox[1])+int(bbox[3]))/2)
          cord_arr[track.track_id].append(((2 * 3.14 * 180)/(cord_arr[track.track_id][0] + cord_arr[track.track_id][1] * 360) * 1000 + 3))
          cord_arr[track.track_id][0] = cord_arr[track.track_id][0]/1920
          cord_arr[track.track_id][1] = cord_arr[track.track_id][1]/1080
          cord_arr[track.track_id][2] = cord_arr[track.track_id][2]/28

        kick_list = []
        for track in cord_arr.keys():
            for track2 in cord_arr.keys():
                if track != track2:
                    x = abs(cord_arr[track][0] - cord_arr[track2][0])
                    y = abs(cord_arr[track][1] - cord_arr[track2][1])
                    if x < 0.0208 and y < 0.0370:
                        kick_list.append([track, track2])
        kick_final = []
        for kick in kick_list:
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1 or track.get_class()!='Person' or track.track_id == kick[0]:
                    continue
                for track2 in tracker.tracks:
                    if not track2.is_confirmed() or track2.time_since_update > 1 or track2.get_class()!='Person' or track2.track_id == kick[1]:
                        continue
                    bbox1 = track.to_tlbr()
                    bbox2 = track2.to_tlbr()
                    if int(bbox1[0]) < int(bbox2[0]) and int(bbox2[2]) < int(bbox1[2]):
                        if track2.track_id not in kick_final:
                            kick_final.append(track2.track_id)
                    else:
                        if track.track_id not in kick_final:
                            kick_final.append(track.track_id)

        for index, track in enumerate(tracker.tracks):
            if track.track_id in kick_final:
                tracker.tracks.pop(index)

        # Compute L2 Norms of all tracked objects
        norm_dict = dict()
        for track1 in tracker.tracks:
          if not track1.is_confirmed() or track1.time_since_update > 1 or track1.get_class()!='Person':
            continue
          norm_dict[track1.track_id] = []
          for track2 in tracker.tracks:
            if not track2.is_confirmed() or track2.time_since_update > 1 or track2.get_class()!='Person':
              continue
            norm_dict[track1.track_id].append(dist.euclidean(cord_arr[track1.track_id],cord_arr[track2.track_id]))
        
        #Create violations list 
        violate_list_id = []
        for key in norm_dict:
          for count, val in enumerate(norm_dict[key]):
            if val == 0.0:
              continue
            else:
                if val < ((thresh-90)/170):
                    violate_list_id.append(key)
                    break
  
        cur_violations = len(violate_list_id)

        #generate predictions for facemasks
        test_data = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1 or track.get_class()=='Person':
                continue  
            bbox = track.to_tlbr()
            try:
              cr_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
              cr_image = cv2.cvtColor(cr_image, cv2.COLOR_RGB2BGR)
              cr_image = cv2.resize(cr_image, (32,32), interpolation = cv2.INTER_AREA)
              test_data.append(cr_image)
            except:
              test_data.append(np.array([[[1]*3]*32]*32))
        scores_facemask = []
        if len(test_data) != 0:
            test_data = np.array(test_data, dtype="float") / 255.0                    
            scores_facemask.append(model.predict(test_data))
        result_thread.append([tracker, scores_facemask, violate_list_id, cur_violations, colors])

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    #video_path = FLAGS.video
    video_path = easygui.fileopenbox()

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0

    #load facemask model
    #model = load_model("./model_data/face_keras_new.h5")
    base_model = tf.keras.applications.MobileNet(
        input_shape=(128,128,3),
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=2,
        classifier_activation="sigmoid"
    )
    base_model.trainable = False
    inputs = keras.Input(shape=(128, 128, 3))

    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout

    outputs = keras.layers.Dense(2)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()])
    model.load_weights('./model_data/me2.h5')

    
    # while video is running
    cv2.namedWindow("Absaar",flags= cv2.WINDOW_GUI_EXPANDED)
    f = open("config.txt", "r")
    thresh = int(f.readline())
    f.close()
    size_var=0
    while True:
        global running_thread, result_thread, hist_thread
        start_time = time.time()
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        if len(result_thread) == 0 and running_thread == False:
            t1 = threading.Thread(target=threader, args = (frame, image, input_size, infer, encoder, nms_max_overlap, tracker, model, thresh,))
            t1.start()
            running_thread = True

        if len(result_thread)!=0:
            index = 0
            # update tracks
            for track in result_thread[0][0].tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()
                
                if class_name != 'Person':
                  face_mask_title = ''
                  if np.argmax(result_thread[0][1][0][index]) == 1:
                    face_mask_title = "without_mask"
                  else:
                    face_mask_title = "with_mask"
                  index +=1
            # draw bbox on screen
                #if int(track.track_id) == 13:
                   #face_mask_title = "without_mask"
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                if track.track_id in result_thread[0][2]:
                  color = (255,0,0)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                if class_name != 'Person':
                  cv2.putText(frame, face_mask_title + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                else:
                    cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            cur_violations = int(result_thread[0][3])
            cv2.putText(frame, "Social distance violators: " + str(cur_violations), (20,120),2,1,(255,255,255),2)
            cv2.putText(frame, "Alert level: ", (20,170),2,1,(255,255,255),2)
            if cur_violations == 0:
                cv2.putText(frame, 'None', (240,170),2,1,(50,205,50),2)
            elif cur_violations < 3:
                cv2.putText(frame, 'Medium', (240,170),2,1,(255,140,0),2)
            else:
                if size_var ==0:
                    cv2.putText(frame, 'HIGH!!', (240,170),2,2,(255,0,0),2)
                    size_var = 1
                else:
                    cv2.putText(frame, 'HIGH!!', (240,170),2,1,(255,0,0),2)
                    size_var = 0
            # calculate frames per second of running detections
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Absaar", result)

            hist_thread = copy.deepcopy(result_thread)
            result_thread = []
            running_thread = False
            
            # if output flag is set, save video file
            if FLAGS.output:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        else:
            if len(hist_thread)!=0:
                index = 0
                # update tracks
                for track in hist_thread[0][0].tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    class_name = track.get_class()

                    if class_name != 'Person':
                      face_mask_title = ''
                      if np.argmax(hist_thread[0][1][0][index]) == 1:
                        face_mask_title = "without_mask"
                      else:
                        face_mask_title = "with_mask"
                      index +=1
                # draw bbox on screen
                    #if int(track.track_id) == 13:
                       #face_mask_title = "without_mask"
                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    if track.track_id in hist_thread[0][2]:
                      color = (255,0,0)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    if class_name != 'Person':
                      cv2.putText(frame, face_mask_title + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    else:
                        cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                cur_violations = int(hist_thread[0][3])
                cv2.putText(frame, "Social distance violators: " + str(cur_violations), (20,120),2,1,(255,255,255),2)
                cv2.putText(frame, "Alert level: ", (20,170),2,1,(255,255,255),2)
                if cur_violations == 0:
                    cv2.putText(frame, 'None', (240,170),2,1,(50,205,50),2)
                elif cur_violations < 3:
                    cv2.putText(frame, 'Medium', (240,170),2,1,(255,140,0),2)
                else:
                    if size_var ==0:
                        cv2.putText(frame, 'HIGH!!', (240,170),2,2,(255,0,0),2)
                        size_var = 1
                    else:
                        cv2.putText(frame, 'HIGH!!', (240,170),2,1,(255,0,0),2)
                        size_var = 0
                # calculate frames per second of running detections
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Absaar", result)
                
                # if output flag is set, save video file
                if FLAGS.output:
                    out.write(result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            else: 
                # calculate frames per second of running detections
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Absaar", result)
                # if output flag is set, save video file
                if FLAGS.output:
                    out.write(result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        fps = 1.0 / (time.time() - start_time + 0.0000001)
        print("FPS: %.2f" % fps)
    cv2.destroyAllWindows()

def runner():
    app.run(main)
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
