import cv2
from cv2 import COLOR_BGR2GRAY


first_frame = None

# video = cv2.VideoCapture(0) # <= Original line
video = cv2.VideoCapture("https://192.168.0.27:8080/video")


while True:
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None: 
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)

    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,0), 3)


    re_gray = cv2.resize(gray, (600, 350))
    re = cv2.resize(delta_frame, (600, 350))
    re_thresh = cv2.resize(thresh_frame, (600, 350))
    re_frame = cv2.resize(frame, (600, 350))

    cv2.imshow("Frame Gray", re_gray)
    cv2.imshow("Delta Frame", re)
    cv2.imshow("Thresh Frame", re_thresh)
    cv2.imshow("Contour", re_frame)

    key = cv2.waitKey(1)
    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows
