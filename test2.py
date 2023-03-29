from Detector import *
import asyncio
import cv2
import numpy as np
import time
import websockets

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz"

classFile = "coco.names"
imagePath = "test/2.jpg"
#videoPath = "D:\OneDrive - Malta College of Arts, Science & Technology\Videos\Advanced Computer Vision with Python - Full Course (1080p_30fps_H264-128kbit_AAC).mp4"
videoPath = 0 # for webcam
threshold = 0.5


#detector.predictImage(imagePath, threshold)

async def receive_frames():
    async with websockets.connect('ws://100.77.189.76:8765') as websocket:
        #cap = cv2.VideoCapture(0)
        start_time = time.time()
        frames = 0
        
        detector = Detector()
        detector.readClasses(classFile)
        detector.downloadModel(modelURL)
        detector.loadModel()

        while True:
            # Receive the frame from the websocket
            data = await websocket.recv()

            # Convert the bytes to a NumPy array
            buffer = np.frombuffer(data, dtype=np.uint8)

            # Decode the JPEG image
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

            # Calculate the elapsed time and FPS
            elapsed_time = time.time() - start_time
            fps = frames / elapsed_time

            # Display the FPS counter on the image
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the image
            #cv2.imshow('Frame', frame)           
            
            detector.predictVideo(frame,threshold)
            
            # Close all windows when q pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):                
                break

            # Increment the frame counter
            frames += 1
            
        cv2.destroyAllWindows()

# Start the client to receive frames
#asyncio.get_event_loop().run_until_complete(receive_frames())