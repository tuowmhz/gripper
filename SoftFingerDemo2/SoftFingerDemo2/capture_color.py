import pyrealsense2 as rs
import numpy as np
import cv2

def capture_color_frame():
    # Configure the pipeline without depth
    pipeline = rs.pipeline()
    config = rs.config()
    # Enable only the color stream at 640x480, 30 fps (you can adjust as needed)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Allow the camera to warm up by discarding some frames
    for _ in range(30):
        frames = pipeline.wait_for_frames()

    # Retrieve a single set of frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Stop the pipeline
    pipeline.stop()

    # Check if the color frame is valid
    if not color_frame:
        raise RuntimeError("Could not acquire color frame.")

    # Convert to numpy array (shape: [480, 640, 3])
    color_image = np.asanyarray(color_frame.get_data())
    return color_image


if __name__ == "__main__":
    # Capture one color frame
    color_image = capture_color_frame()
    
    # Save the image
    cv2.imwrite("color_only.png", color_image)
    print("Saved 2D color image as 'color_only.png'.")

    # (Optional) display the image
    cv2.imshow("Color Frame", color_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
