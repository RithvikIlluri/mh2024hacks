import cv2
saved_data = {}
size = 0
temp = []

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return [faces,w,h]

count = 1
print("camera open!")

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(video_frame)[0]
    aspect_ratio = detect_bounding_box(video_frame)[1]/detect_bounding_box(video_frame)[2]

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    k = cv2.waitKey(100) & 0xff
    if k == 5:
        break
    elif count >= 15:
        break
    count += 1
    print(count)

    if count==7:
        temp.append(aspect_ratio)
    

video_capture.release()
cv2.destroyAllWindows()
print("camera closed!")
name = input("Enter your name: ")
print(size)
temp.append(name)
saved_data[str(temp[0])] = temp[1]

print(saved_data)