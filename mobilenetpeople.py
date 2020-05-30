
import cv2
import numpy as np
import urllib


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
  
  
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

def getPredictions(frame):
    
    nparr = np.fromstring(frame.data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    
    (H, W) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if (confidence > 0.5):

            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue
            
            box = detections[0, 0, i, 3:7] *  np.array([W, H, W, H])
            
            print(box)
            
    return box