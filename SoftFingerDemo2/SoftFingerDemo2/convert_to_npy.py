import cv2
import numpy as np
import os


def png_to_npy(image_path="refined_mask.png", output_path="refined_mask.npy"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(
            "Failed to load image. It may be corrupted or an unsupported format."
        )

    print(f"Original image shape: {image.shape}")

    # Convert to RGB
    if image.ndim == 2:
        print("Image is grayscale")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 4:
        print("Image has alpha channel")
        alpha = image[:, :, 3]
        mask = alpha > 0
        bg = np.ones_like(image[:, :, :3], dtype=np.uint8) * 255
        image_rgb = np.where(mask[:, :, None], image[:, :, :3], bg)
    elif image.shape[-1] == 3:
        print("Image is already RGB")
        image_rgb = image
    else:
        raise ValueError("Unsupported image format")

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Save debug binary image
    cv2.imwrite("debug_binary.png", binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")

    if not contours:
        raise ValueError("No object found in image")

    x, y, w, h = cv2.boundingRect(contours[0])
    print(f"Cropping to bounding box: x={x}, y={y}, w={w}, h={h}")
    cropped = image_rgb[y : y + h, x : x + w]

    scale = min(32 / w, 32 / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    print(f"Resizing to: {new_w}x{new_h}")
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.ones((32, 32, 3), dtype=np.uint8) * 255
    x_offset = (32 - new_w) // 2
    y_offset = (32 - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    formatted = np.transpose(canvas, (2, 0, 1)).reshape(1, 3, 32, 32)
    np.save(output_path, {"image": formatted})

    assert os.path.exists(output_path), "Save failed unexpectedly!"
    print(f"✅ Saved .npy file to: {output_path}")


print("🔁 convert_to_npy.py started")

try:
    png_to_npy("refined_mask.png", "refined_mask.npy")
    print("✅ convert_to_npy.py completed")
except Exception as e:
    print(f"❌ convert_to_npy.py failed: {e}")


# # Run it
# png_to_npy("refined_mask.png", "refined_mask.npy")
