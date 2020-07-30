from flask import Flask, jsonify, request
import face_recognition
import torch
import urllib
import cv2
import numpy as np
import os

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
# warnings.filterwarnings('ignore')

print("[File]: ",torch.__file__)
print("[Version]: ",torch.__version__)

model_dir = "./resources/anti_spoof_models"

model_test = AntiSpoofPredict()
image_cropper = CropImage()

server = Flask(__name__)

@server.route("/", methods = ["POST"])
def func():
    data = request.json
    urls = list(data["urls"])
    unknown_url = data["unknown_url"]

    faces = 0
    label = ''

    resp = urllib.request.urlopen(unknown_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = frame


    image_bbox, faces = model_test.get_bbox(image)
    
    if(faces==1):
        prediction = np.zeros((1, 3))

        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))

        l = np.argmax(prediction)
        value = prediction[0][l]/2
        if l == 1:
            label = 'true'
        else:
            label = 'fake'

    error = -1
    body = "False"
    message = ""

    if(faces==0):
        error = 1
        body = "Error"
        message = "No Face Detected"
    elif(faces>1):
        error = 2
        body = "Error"
        message = "Multiple Faces Detected"
    elif(faces==1 and label == 'fake'):
        error = 3
        body = "Error"
        message = "Only 1 Face Detected but is fake"

    elif(faces==1 and label == 'true'):

        boxes = face_recognition.face_locations(image)

        if(len(boxes)==0):
            message = "No Face Detected"
            error = 1
            body = "Error"

            # return False, error, msg

        else:
            unknown = face_recognition.face_encodings(image,boxes)[0]

            for im_path in urls:
                resp = urllib.request.urlopen(im_path)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                boxes = face_recognition.face_locations(image)
                if(len(boxes)==0):
                    continue

                known = face_recognition.face_encodings(image,boxes)[0]
                matches = face_recognition.compare_faces([unknown], known)
                if(matches[0]==True):
                    error = -1
                    message = "Validated"
                    body = "True"
                    break

            if(body!="True"):
                error = 4
                message = "Face not Validated"
                body = "Error"

    return jsonify(statusCode = 200, body= body,error= error,message= message)

if __name__ == "__main__":
    server.run(host='0.0.0.0', port = 80)