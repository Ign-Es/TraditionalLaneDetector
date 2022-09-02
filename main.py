import tld
import cv2
import numpy as np

# Set hsv ranges for colors
yellow_range = {'low': [25, 140, 100], 'high': [45, 255, 255]}
white_range = {'low': [0, 0, 150], 'high': [180, 100, 255]}
red_range = {'low_1': [0,140,100], 'high_1': [15,255,255], 'low_2': [165,140,100], 'high_2': [180,255,255]}
colors = {"red": red_range, "yellow": yellow_range, "white": white_range}
color_ranges = {color: tld.ColorRange.fromDict(d) for color, d in list(colors.items())}

# Initialize detector
detector = tld.LineDetector()

#cap = cv2.VideoCapture("test2.mp4")
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    _, image = cap.read()

    # Resize the image to the desired dimensions
    height_original, width_original = image.shape[0:2]
    img_size = (width_original, height_original)
    top_cutoff = int(height_original * 0.4)
    if img_size[0] != width_original or img_size[1] != height_original:
        image = cv2.resize(image, img_size, interpolation=cv2.INTER_NEAREST)
    image = image[top_cutoff:, :, :]

    # Extract the line segments for every color
    detector.setImage(image)
    detections = {
        color: detector.detectLines(ranges) for color, ranges in list(color_ranges.items())
    }

    # Remove the offset in coordinates coming from the removing of the top part and
    arr_cutoff = np.array([0, top_cutoff, 0, top_cutoff])
    arr_ratio = np.array(
        [
            1.0 / img_size[1],
            1.0 / img_size[0],
            1.0 / img_size[1],
            1.0 / img_size[0],
        ]
    )
    # Plot Segments
    colorrange_detections = {color_ranges[c]: det for c, det in list(detections.items())}
    debug_img = tld.plotSegments(image, colorrange_detections)
    cv2.imshow("result", debug_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

