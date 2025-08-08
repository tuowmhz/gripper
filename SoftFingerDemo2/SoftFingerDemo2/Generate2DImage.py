import cv2
import numpy as np
import matplotlib.pyplot as plt

def postprocess_mask(mask, scale=1.5):
    """
    Example postprocessing that resizes the mask by a factor and draws some shape overlays.
    """
    # Convert single-channel mask to BGR for visualization
    colored_mask = np.stack([mask*255, mask*255, mask*255], axis=-1).astype(np.uint8)
    
    # Resize with nearest-neighbor to keep edges crisp
    new_size = (int(colored_mask.shape[1]*scale), int(colored_mask.shape[0]*scale))
    resized_mask = cv2.resize(colored_mask, new_size, interpolation=cv2.INTER_NEAREST)

    # Now draw any shape on top if you want
    # For instance, draw a rectangle for demonstration:
    cv2.rectangle(resized_mask, (10, 10), (50, 50), (0, 0, 255), 2)
    return resized_mask

if __name__ == "__main__":
    refined_mask = cv2.imread("refined_mask.png", cv2.IMREAD_GRAYSCALE)
    # threshold just in case it is not purely 0/255
    refined_mask_bin = (refined_mask > 127).astype(np.uint8)

    final_img = postprocess_mask(refined_mask_bin, scale=1.5)
    cv2.imwrite("final_result.png", final_img)
    print("Saved postprocessed result as final_result.png")

    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.title("Postprocessed Mask")
    plt.show()
