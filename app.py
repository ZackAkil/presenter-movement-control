# import time
import pyautogui


# time.sleep(5)
# for i in range(1,10):
#     time.sleep(1)
#     pyautogui.press(str(i))
#     pyautogui.press('enter')
# print('finnish')

def goto_slides_number(n):
    pyautogui.press(str(n))
    pyautogui.press('enter')


# from SimpleCV import Image, Camera

# cam = Camera()
# img = cam.getImage()
# img.save("filename.jpg")

import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2.service_account import Credentials


import numpy as np
import cv2

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('slides', metavar='n', type=int,
                    help='number of slides')

args = parser.parse_args()

if args.slides:
    number_of_slides = args.slides
    print(number_of_slides, 'slides')
else:
    print('spedicify number of slides')
    exit()

# Instantiates a client
creds = Credentials.from_service_account_file("vision.json")
client = vision.ImageAnnotatorClient(credentials=creds)


def get_prediction(content):

    image = types.Image(content=content)
    # Performs label detection on the image file
    response = client.object_localization(image=image)
    persons = [a for a in response.localized_object_annotations if a.name=="Person"]
    if len(persons)==1:
        person = persons[0]
        left = person.bounding_poly.normalized_vertices[0].x
        right = person.bounding_poly.normalized_vertices[1].x
        return np.mean([left,right])
    else:
        return None
    # labels = response.label_annotations

    # print('Labels:')
    # for label in labels:
    #     print(label.description)


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = np.flip(frame, 1)

    _, im_buf_arr = cv2.imencode(".jpg", frame)
    content = im_buf_arr.tobytes()
    pos = get_prediction(content)

    for i in range(number_of_slides):
        frame[:,int((frame.shape[1]/number_of_slides)*i),:] = 0

    if pos:
        frame[:,int(frame.shape[1]*pos),:] = 255
        slide_to_goto = int((pos) * number_of_slides)+1
        print('going to ', slide_to_goto)
        goto_slides_number(slide_to_goto)
        



    # frame

    # # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()






