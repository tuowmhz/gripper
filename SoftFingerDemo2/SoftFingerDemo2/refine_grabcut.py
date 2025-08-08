# import cv2
# import numpy as np

# def refine_bounding_box_grabcut(image, box, iterations=5):
#     """
#     image: BGR image (numpy array)
#     box: (x, y, w, h)
#     """
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     bgdModel = np.zeros((1, 65), np.float64)
#     fgdModel = np.zeros((1, 65), np.float64)

#     x, y, w, h = box
#     rect = (x, y, w, h)

#     # Run GrabCut
#     cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)

#     # Where mask=2 or 0 => background, otherwise foreground
#     refined_mask = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    
#     # Create final segmented image
#     segmented = image * refined_mask[:, :, np.newaxis]
#     return refined_mask, segmented

# if __name__ == "__main__":
#     color_image = cv2.imread("color_only.png")
#     box = (100, 120, 80, 100)  # Example bounding box from YOLO
#     mask, segmented = refine_bounding_box_grabcut(color_image, box)

#     cv2.imwrite("refined_mask.png", mask*255)
#     cv2.imwrite("segmented.png", segmented)
#     print("Saved refined mask and segmented object.")

import cv2
import numpy as np
import sys

def refine_bounding_box_grabcut(image, box, iterations=5):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    x, y, w, h = box
    rect = (x, y, w, h)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
    refined_mask = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
    segmented = image * refined_mask[:, :, np.newaxis]
    return refined_mask, segmented

if __name__ == "__main__":
    # 1) Read the bounding box from bbox.txt
    try:
        with open("color_bbox.txt", "r") as f:
            line = f.read().strip()
    except FileNotFoundError:
        print("No bbox.txt found. Did you run yolo_detection.py first?")
        sys.exit(1)

    if not line:
        print("No bounding box data (bbox.txt is empty). Exiting.")
        sys.exit(0)

    # e.g. line = "150 0 259 166"
    x_str, y_str, w_str, h_str = line.split()
    x, y, w, h = int(x_str), int(y_str), int(w_str), int(h_str)

    # 2) Read the image
    color_image = cv2.imread("color_only.png")
    if color_image is None:
        raise FileNotFoundError("Could not read 'color_only.png'. Check the file name/path.")

    # 3) Run GrabCut with the bounding box
    mask, segmented = refine_bounding_box_grabcut(color_image, (x, y, w, h))

    inverted_mask = 1 - mask

    cv2.imwrite("refined_mask.png", inverted_mask*255)
    cv2.imwrite("segmented.png", segmented)
    print("Saved refined mask and segmented object.")
