import sys
import cv2
import numpy as np

# main(sys.argv[1:])
window_name = 'Paul Mello'

imageName = input("File Name: ")

src = cv2.imread(imageName, cv2.IMREAD_COLOR)

if src is None:
    print('Error opening image!')
    print('Usage: pdisplay.py image_name\n')

img = cv2.resize(src, (300, 590))

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canned = cv2.Canny(grey, 100, 200)

ind = 0

while True:
    cv2.imshow(window_name + ", Original Image", img)
    cv2.imshow(window_name + ", Grey-Scaled Image", grey)
    cv2.imshow(window_name + ", Canny Edge Detection", canned)

    c = cv2.waitKey(500)
    if c == 27:  # ESC
        break

    ind += 1

