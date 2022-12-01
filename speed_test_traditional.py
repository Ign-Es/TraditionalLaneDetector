import cv2
import tld
import time
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

video_name = "testing_video2"
video_format = ".avi"
video_path = "videos/"+video_name+video_format
# Initialize video
cap = cv2.VideoCapture(video_path)


# Set hsv ranges for colors
yellow_range = {'low': [25, 140, 100], 'high': [45, 255, 255]}
white_range = {'low': [0, 0, 150], 'high': [180, 100, 255]}
red_range = {'low_1': [0,140,100], 'high_1': [15,255,255], 'low_2': [165,140,100], 'high_2': [180,255,255]}
#colors = {"red": red_range, "yellow": yellow_range, "white": white_range}
#colors = {"white":white_range, "yellow": yellow_range}
colors = {"white":white_range}
color_ranges = {color: tld.ColorRange.fromDict(d) for color, d in list(colors.items())}

# Initialize traditional detector
detector = tld.LineDetector()

counter = 0
latency_list = []
while cap.isOpened():
    try:
        # Read frame from the video
        ret, frame = cap.read()
    except:
        continue

    if ret:


        # Resize the image to the desired dimensions
        height_original, width_original = frame.shape[0:2]
        img_size = (800, 288)
        top_cutoff = int(height_original * 0.)
        if img_size[0] != width_original or img_size[1] != height_original:
            frame = cv2.resize(frame, img_size, interpolation=cv2.INTER_NEAREST)
        frame = frame[top_cutoff:, :, :]

        # Extract the line segments for every color
        start = time.time()
        detector.setImage(frame)
        detections = {
            color: detector.detectLines(ranges) for color, ranges in list(color_ranges.items())
        }
        end = time.time()
        latency_list.append(end-start)
        counter += 1
        if counter > 101:
            break
    else:
        break

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()

latency_list = np.asarray(latency_list[1:-1])*1000
print(latency_list)
print(len(latency_list))
print("The mean time of execution of above program is :",
      mean(latency_list), "ms")
print("The maximum time of execution of above program is :",
      max(latency_list), "ms")
print("The minimum time of execution of above program is :",
      min(latency_list), "ms")

plt.figure(0)
plt.plot(range(len(latency_list)), latency_list, linestyle=':', marker='.',  c='blue', mfc='red')
plt.xlim([0, len(latency_list)])
plt.ylim([min(latency_list)-1, max(latency_list)+1])
plt.ylabel('Time[ms]')
plt.xlabel('Iteration')
#pyplot.show()
plt.savefig('latency_tld1_'+video_name+'.png', dpi=300)