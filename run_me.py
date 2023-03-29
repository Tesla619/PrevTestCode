from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz"

classFile = "coco.names"
imagePath = "test/2.jpg"
#videoPath = "D:\OneDrive - Malta College of Arts, Science & Technology\Videos\Advanced Computer Vision with Python - Full Course (1080p_30fps_H264-128kbit_AAC).mp4"
videoPath = 1 # for webcam
threshold = 0.5

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
#detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath,threshold)
