#!/bin/bash
set -e  # stop on error

echo "Step 1: Capturing color frame..."
python3 capture_color.py  # saves color_only.png

echo "Step 2: Running YOLO detection..."
python3 yolo_detection.py  # saves bbox to bbox.txt

echo "Step 3: Running color detection..."
python3 color_detection.py  # saves bbox to bbox.txt

echo "Step 4: Refining bounding box with GrabCut..."
python3 refine_grabcut.py

echo "Step 5: Converting the png file to npy..."
python3 convert_to_npy.py


# #!/bin/bash
# # Exit immediately if any command returns a non-zero status
# set -e

# # ---------------------------------------------------------
# # (Optional) Activate your Python virtual environment here.
# # Comment out if you don't use a virtualenv:
# # source /path/to/your/venv/bin/activate
# # ---------------------------------------------------------

# # 1) Capture a color frame from the RealSense camera
# echo "Step 1: Capturing color frame..."
# python3 capture_color.py

# # 2) Run your YOLO detection (example script name)
# echo "Step 2: Running YOLO detection..."
# python3 yolo_detection.py

# # 3) Refine the bounding box with GrabCut (example script name)
# echo "Step 3: Refining bounding box with GrabCut..."
# python3 refine_grabcut.py

# # 4) (Optional) Run any additional postprocessing (e.g. PointNet++)
# # echo "Step 4: Running postprocessing..."
# # python3 postprocess_pointcloud.py

# # ---------------------------------------------------------
# # (Optional) Deactivate the virtual environment
# # deactivate
# # ---------------------------------------------------------

# echo "Pipeline completed successfully!"
