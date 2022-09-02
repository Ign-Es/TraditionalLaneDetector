import tld
import cv2
import numpy as np

'''
colors:
  RED:
    low_1: [0,140,100]
    high_1: [15,255,255]
    low_2: [165,140,100]
    high_2: [180,255,255]
  WHITE:
    low: [0,0,150]
    high: [180,100,255]
  YELLOW:
    low: [25,140,100]
    high: [45,255,255]
'''

colors = { 'low_1': [0,0,150], 'high_1': [180,100,255], 'low_2': [25,140,100], 'high_2': [45,255,255] }
color_range = tld.ColorRange.fromDict(colors)
im = cv2.imread('../test_image.png')
detector = tld.LineDetector()
detector.setImage(im)
detections = detector.detectLines(color_range)
im2 = tld.plotSegments(im, {color_range: detections})
cv2.imshow("result", im2)
cv2.waitKey(0)
im2 = tld.plotMaps(im, {color_range: detections})
cv2.imshow("result", im2)
cv2.waitKey(0)
