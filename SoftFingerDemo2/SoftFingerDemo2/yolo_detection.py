import cv2
import numpy as np

def load_yolo_model(weights_path, config_path, class_names_path):
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return net, class_names

def detect_objects_yolo(net, class_names, image, conf_threshold=0.35, nms_threshold=0.4):
    (H, W) = image.shape[:2]
    # YOLOv3 image size is often 416x416 or 608x608, but you can adapt
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layer_indices = net.getUnconnectedOutLayers()
    output_names = [layer_names[i - 1] for i in output_layer_indices]
    outputs = net.forward(output_names)

    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Non-maxima suppression to refine overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    final_boxes = []
    final_classIDs = []
    final_confidences = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            final_boxes.append(boxes[i])
            final_classIDs.append(classIDs[i])
            final_confidences.append(confidences[i])

    return final_boxes, final_classIDs, final_confidences

# if __name__ == "__main__":
#     # Example usage
#     color_image = cv2.imread("color_only.png")
#     net, class_names = load_yolo_model(
#         "yolov3.weights",
#         "yolov3.cfg",
#         "coco.names"
#     )
#     boxes, classIDs, confs = detect_objects_yolo(net, class_names, color_image)
#     print("Detections:", boxes, classIDs, confs)

#     # Visualize
#     for (box, cls, conf) in zip(boxes, classIDs, confs):
#         x, y, w, h = box
#         label = f"{class_names[cls]}: {conf:.2f}"
#         cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(color_image, label, (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     cv2.imwrite("detections.png", color_image)
#     print("Saved detection result as detections.png")
if __name__ == "__main__":
    color_image = cv2.imread("color_only.png")
    net, class_names = load_yolo_model("yolov3.weights", "yolov3.cfg", "coco.names")
    boxes, classIDs, confs = detect_objects_yolo(net, class_names, color_image)
    print("Detections:", boxes, classIDs, confs)
    
    # Visualize
    for (box, cls, conf) in zip(boxes, classIDs, confs):
        x, y, w, h = box
        label = f"{class_names[cls]}: {conf:.2f}"
        cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(color_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("detections.png", color_image)
    print("Saved detection result as detections.png")

    # Choose the first bounding box to refine
    if len(boxes) > 0:
        x, y, w, h = boxes[0]
        # Write the bounding box to a simple text file
        with open("bbox.txt", "w") as f:
            f.write(f"{x} {y} {w} {h}")
    else:
        # Write an empty file to signal no detection
        with open("bbox.txt", "w") as f:
            f.write("")
