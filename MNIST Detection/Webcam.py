import sys
import cv2

from keras.models import load_model

# READ NOTES BELOW
# Comment / Modify Two lines below depending on desired Prerecorded or Live, only 1 can be uncommented at a time

#vid = cv2.VideoCapture("Data/MNIST_Video.mp4")     # Adjust This File Path for Prerecorded video processing
vid = cv2.VideoCapture(0)                           # Live Capture of Webcam

framePercent = 0.8  # Moderately Helps Control Frame Rate

model = load_model("paul_mello_7025_p1.h5")
model.compile(optimizer = "Adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])

processedVid = cv2.VideoWriter("ProcessedMNISTWebcam.avi", cv2.VideoWriter_fourcc(*'MJPG'), 10,
                               (int(vid.get(3) * framePercent) , int(vid.get(4)* framePercent))) # Size of Frame

while True:
    hull = []

    ret, frame = vid.read()

    width = int(frame.shape[1] * framePercent)
    height = int(frame.shape[0] * framePercent)
    frame = cv2.resize(frame, (width, height))
    srcFrame = frame.copy()

    #frame = cv2.resize(frame, (500,500))

    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    BinaryImage = cv2.threshold(grayScale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contour = cv2.findContours(BinaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    CannedImage = cv2.Canny(grayScale, 40, 80)

    contour = contour[0] if len(contour) == 2 else contour[1]
    contour = sorted(contour, key = lambda x: cv2.boundingRect(x)[0])

    for c in contour:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 200:
            box = BinaryImage[y: y + h, x: x + w]
            box = cv2.resize(box, (28, 28), interpolation = cv2.INTER_AREA)
            box = box.reshape(1, 28, 28, 1)
            prediction = model.predict(box)
            label = prediction.argmax()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(frame, str(label), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    #cv2.imshow("Paul-Jason Mello SrcFrame", srcFrame)
    #cv2.imshow("Mello Binary Frame", BinaryImage)
    #cv2.imshow("Mello Canny Frame", CannedImage)
    cv2.imshow("Mello Live MNIST", frame)
    processedVid.write(frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

vid.release()
processedVid.release()
cv2.destroyAllWindows()
