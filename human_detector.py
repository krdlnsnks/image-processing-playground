import cv2

def detect_human(image_path):
    # Load the pre-trained HOG descriptor for human detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Load the image
    image = cv2.imread(image_path)

    # Detect humans in the image
    (human_rects, _) = hog.detectMultiScale(image, winStride=(4, 4),
                                             padding=(8, 8), scale=1.05)

    # Draw rectangles around the detected humans
    for (x, y, w, h) in human_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Human Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the path to the image
image_path = "rect.png"
detect_human(image_path)