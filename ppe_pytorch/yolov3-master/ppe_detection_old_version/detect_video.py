import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes'      , './data/ppe_voc.names'           , 'path to classes file')
flags.DEFINE_string('weights'      , './checkpoint/yolov3_train_60.tf', 'path to weights file')
flags.DEFINE_string('image'        , './data/girl.png'                , 'path to input image')
flags.DEFINE_string('tfrecord'     , None                             , 'tfrecord instead of image')
# flags.DEFINE_string('output'      , './output.jpg'                   , 'path to output image')
flags.DEFINE_string('video'        , './video/test1.mp4'               , 'path to video file or number for webcam)')
flags.DEFINE_string('output'       , './video_output.avi'                             , 'path to output video')
flags.DEFINE_string('output_format', 'XVID'                           , 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('tiny'        , False                            , 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size'        , 416                              , 'resize images to')
flags.DEFINE_integer('num_classes' , 3                                , 'number of classes in the model')


    
def main(_argv): 
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    model = YoloV3(classes=FLAGS.num_classes)
    model.load_weights(FLAGS.weights).expect_partial()
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('weights loaded')
    logging.info('classes loaded')

    # TIME
    # times = []
    # t1 = time.time()
    # t2 = time.time()
    # times.append(t2-t1)
    # times = times[-20:]

    # FRAME 
    # frame = 1 
    # frame = frame + 1


    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))


    while True:
        status, img = vid.read()
        if status is None:
            break
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        # Detector part
        boxes, scores, classes, nums = model.predict(img_in)


        # Decision part
        # ...
        # ...


        # Drawing part
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "Frame ...", (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)



        # Write video output part
        if FLAGS.output:
            out.write(img)

        # cv2.imshow('output', img)
        # if cv2.waitKey(1) == ord('q'):
        #     break


        # Camera 
        # if FLAGS.camera:
            # open the camera 
            # pass


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit: 
        pass


# Directory and activate environments
# cd codiadata/pong/ppe_detection/ppe_detection_tf2 && conda activate yolov3-tf2-gpu

# Command for run
# python new.py --vdo test --save test --classes ./data/ppe_voc.names --weights ./checkpoint/yolov3_train_60.tf

# python detect_video.py \--classes ./data/ppe_voc.names \--num_classes 3 \--weights ./checkpoint/yolov3_train_60.tf \--video ./video/test1.mp4