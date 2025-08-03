import cv2
import numpy as np

# Load the image
img = cv2.imread('f.jpg') 
if img is None:
    print("Error: Image not found. Please check the file name and path.")
else:
    # Resize the image
    img = cv2.resize(img, (400, 400))

    # Define filters
    identity_kernel = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    sharpen_kernel = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ])

    edge_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    box_blur_kernel = np.ones((3, 3), np.float32) / 9

identity = cv2.filter2D(img, -1, identity_kernel)
sharpened = cv2.filter2D(img, -1, sharpen_kernel)
edge_detected = cv2.filter2D(img, -1, edge_kernel)
box_blur = cv2.filter2D(img, -1, box_blur_kernel)
gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)

# Show images
cv2.imshow("Original", img)
cv2.imshow("Identity", identity)
cv2.imshow("Sharpen", sharpened)
cv2.imshow("Edge Detection", edge_detected)
cv2.imshow("Box Blur", box_blur)
cv2.imshow("Gaussian Blur", gaussian_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()