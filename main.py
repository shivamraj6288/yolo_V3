import numpy as np
import cv2
# import torch
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# print(classes)

layer_names = net.getLayerNames()
# print(layer_names)
# print(net.getUnconnectedOutLayers())

output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
# print(output_layers)

img = cv2.imread("studyroom.jpg")
img = cv2.resize(img, None, fx=0.5, fy=0.5)
height, width, channels = img.shape

# detecting objects

blob = cv2.dnn.blobFromImage(
    img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n), img_blob)

net.setInput(blob)
outs = net.forward(output_layers)
# print(outs)

# showing information on strings

boxes = []
confidences = []
class_ids = []


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # object detected
            center_x = int(detection[0]*width)
            center_y = int(detection[1] * height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            # cv2.circle(img, (center_x, center_y), 50, (0, 255, 0), 10)
            # rectangle coordinates

            x = int(center_x-w/2)
            y = int(center_y-h/2)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # boxes.append([x,y,w,h])
            # confidences.append(confidence)
        else:
            print(confidence, 'zero confidence')


cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
