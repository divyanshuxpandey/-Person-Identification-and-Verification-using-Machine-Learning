import cv2
KNOWN_DISTANCE = 200
KNOWN_WIDTH = 14.3
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def focal_length(measured_distance, real_width, width_in_rf_image):

    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value
def distance_finder(focal_length, real_face_width, face_width_in_frame):

    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance
def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        # cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)
        face_width = w

    return face_width
# this is gender detection
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
padding=20
def faceBox(faceNet, frames):
    frameHeight=frames.shape[0]
    frameWidth=frames.shape[1]
    blob=cv2.dnn.blobFromImage(frames, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frames, (x1,y1),(x2,y2),(0,255,0), 1)
    return frames, bboxs

ref_image = cv2.imread("Ref_image.jpg")

ref_image_face_width = face_data(ref_image)
focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)
print(focal_length_found)
cv2.imshow("ref_image", ref_image)
person_count =0;
while True:
    _, frame = cap.read()
    face_width_in_frame = face_data(frame)
    if face_width_in_frame != 0:
        Distance = distance_finder(focal_length_found, KNOWN_WIDTH, face_width_in_frame)
        if Distance >=20 :
            person_count=1
        cv2.putText(
            frame, f"Distance = {round(Distance)} CM", (50, 50), fonts, 1, (0, 0, 255), 2
        )

    cv2.putText(frame, f"Persons: {person_count}", (40, 700), fonts, 1, (0, 255, 0), 2)


# new gender output code
    frameFace, bboxes = faceBox(faceNet, frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for bbox in bboxes:
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = "{},{}".format(gender, age)
        cv2.rectangle(frameFace, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
    if face_width_in_frame == 0:
        person_count=0

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
