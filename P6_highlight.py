# Lucas de la Pena
# highlight.py
# 12/2/2024

import cv2
import numpy as np
import argparse

# Global variables for rectangle selection
rectangleStart = None
rectangleEnd = None
drawing = False

def draw_rectangle(event, x, y, flags, param):
    global rectangleStart, rectangleEnd, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rectangleStart = (x, y)
        rectangleEnd = rectangleStart
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        rectangleEnd = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangleEnd = (x, y)

def main():
    """
    Main function to run the program.
    """
    global rectangleStart, rectangleEnd

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        prog='highlight',
        description='Highlight and equalize an ROI in an image.'
    )
    parser.add_argument('imagefile', help='Path to the image file.')
    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.imagefile)
    if image is None:
        print("Error: Unable to open image file.")
        return

    print("Click and drag to select a rectangle. Press 'q' to quit.")
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)

    while True:
        # Create a temporary copy of the image for live updates
        tempImage = image.copy()

        # Draw the rectangle dynamically during mouse movement
        if rectangleStart and rectangleEnd:
            x1, y1 = rectangleStart
            x2, y2 = rectangleEnd
            topLeft = (min(x1, x2), min(y1, y2))
            bottomRight = (max(x1, x2), max(y1, y2))
            cv2.rectangle(tempImage, topLeft, bottomRight, (0, 255, 0), 2)

        cv2.imshow("Image", tempImage)

        # Exit loop when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Validate and process the selected rectangle
    if rectangleStart and rectangleEnd:
        x1, y1 = rectangleStart
        x2, y2 = rectangleEnd
        topLeft = (min(x1, x2), min(y1, y2))
        bottomRight = (max(x1, x2), max(y1, y2))

        xStart, yStart = topLeft
        xEnd, yEnd = bottomRight
        roi = image[yStart:yEnd, xStart:xEnd]

        if roi.size == 0:
            print("Invalid ROI. No changes applied.")
            return

        # Dim the entire image
        dimmed_image = (image * 0.75).astype(np.uint8)

        # Convert ROI to grayscale and apply histogram equalization
        roiGray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        equalizedRoi = cv2.equalizeHist(roiGray)
        equalizedRoiColor = cv2.cvtColor(equalizedRoi, cv2.COLOR_GRAY2BGR)

        # Replace the ROI in the dimmed image with the equalized ROI
        dimmed_image[yStart:yEnd, xStart:xEnd] = equalizedRoiColor

        # Display the final image
        cv2.imshow("Final Image", dimmed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Selected ROI: Top-left {topLeft}, Bottom-right {bottomRight}")

if __name__ == '__main__':
    main()
