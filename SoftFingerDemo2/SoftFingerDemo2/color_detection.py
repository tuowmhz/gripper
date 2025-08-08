import cv2
import numpy as np

def detect_colored_objects(image, min_area=300):
    """
    Detect bounding boxes for yellow, blue, and green objects against a black background.
    Returns a list of (x, y, w, h) bounding boxes.
    
    :param image: BGR (OpenCV) image
    :param min_area: minimum area of contour to accept as valid object
    """
    # 1) Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2) Define color ranges in HSV
    #    These are approximate ranges. Adjust them as needed for your lighting / object colors.

    # a) Yellow range
    lower_yellow = (20, 100, 100)
    upper_yellow = (30, 255, 255)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # b) Blue range
    lower_blue = (90, 100, 50)
    upper_blue = (130, 255, 255)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # c) Green range
    lower_green = (40, 70, 50)
    upper_green = (70, 255, 255)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Combine all color masks
    combined_mask = mask_yellow | mask_blue | mask_green

    # 3) Optional morphological ops to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4) Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append((x, y, w, h))
    return bboxes

if __name__ == "__main__":
    # 1) Read the image you saved from RealSense D435i (black background, pure colored objects)
    color_image = cv2.imread("color_only.png")
    if color_image is None:
        raise FileNotFoundError("Could not read 'color_only.png'. Check the file path.")

    # 2) Detect bounding boxes for yellow, blue, and green objects
    boxes = detect_colored_objects(color_image, min_area=300)
    print("Detected bounding boxes:", boxes)

    # 3) Draw the bounding boxes on the image (for visualization)
    for (x, y, w, h) in boxes:
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 4) Save result
    cv2.imwrite("color_detection_result.png", color_image)
    print("Saved color_detection_result.png with bounding boxes.")

    # 5) (Optional) Write bounding boxes to a file for later GrabCut, etc.
    with open("color_bbox.txt", "w") as f:
        for (x, y, w, h) in boxes:
            f.write(f"{x} {y} {w} {h}\n")
    print(f"Saved {len(boxes)} bounding boxes to color_bbox.txt.")

