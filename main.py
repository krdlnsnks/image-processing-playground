import cv2
import numpy as np

def detect_rectangles(image_path):
    # Read the image
    image = cv2.imread(image_path)
    original_image = image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If the contour has 4 vertices, it's likely a rectangle
        if len(approx) == 4:
            # Check if the contour is convex
            if cv2.isContourConvex(approx):
                # Get the bounding rectangle of the contour
                x, y, w, h = cv2.boundingRect(approx)

                # Check if aspect ratio is close to 1 (to filter out non-rectangular shapes)
                aspect_ratio = float(w) / h
                if 0.9 <= aspect_ratio <= 1.1:
                    # Draw the rectangle on the original image
                    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Rectangles Detected", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Provide the path to the image
image_path = "rect.png"
detect_rectangles(image_path)
