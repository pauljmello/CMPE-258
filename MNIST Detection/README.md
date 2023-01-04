# Live Webcam / Prerecorded MNIST Detection

## General Information

Live / Prerecorded MNIST detector. Which takes a video, loads a predefined model, then preprocesses the video for standardization.
Next we create specialized video feeds like binary images, uncanny edges, and countour lines. Using the contours,
we check the area as a threshold to maintain framerate. Then predict the value of the object inside, assuming the contoured area is larger than
200 pixels total. Finally, these predictions are output on a live feed.

## Testing Information
***
Adhere to comments from line 7 - 11 in Webcam.py file.

Essentially, only one line can be commented out at a time between line 10 or 11.
Run Line 11 for Live Webcam detection. Modify the file path for Line 10 then run it for prerecorded video processing.
Everything else in the file should not need to be changed. 

Notes:
Changing uncommented lines does not change the title/file path names for any videos you test.
File names were modified post processing for ease of identification.
Webcam.py is the master file.
