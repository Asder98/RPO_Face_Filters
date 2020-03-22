import numpy as np
import dlib
import cv2
import matplotlib.pyplot as plt
import skimage
from scipy.interpolate import interp1d
from imutils import face_utils
import argparse
import os
from PIL import Image


def drawAllPoints(img, pts):
    overlay = img.copy()
    for i in range(len(pts)):
        #print("x: ", pts[i][0], " y: ", pts[i][1])
        cv2.circle(overlay, (pts[i][0], pts[i][1]), 2, (255, 0, 0), 2)

    return overlay

def Eyeliner(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = face_detector(gray, 0)# The 2nd argument means that we upscale the image by 'x' number of times to detect more faces.
    if bounding_boxes:    
        for i, bb in enumerate(bounding_boxes):
            face_landmark_points = lndMrkDetector(gray, bb)
            face_landmark_points = face_utils.shape_to_np(face_landmark_points)

            op = drawAllPoints(frame, face_landmark_points)

            #eye_landmark_points = getEyeLandmarkPts(face_landmark_points)
            #eyeliner_points = getEyelinerPoints(eye_landmark_points)
            #op = drawEyeliner(frame, eyeliner_points)
        
        return op
    else:
        return frame

def video(src = 0):

    cap = cv2.VideoCapture(src)

    if args['save']:
        if os.path.isfile(args['save']+'.avi'):
            os.remove(args['save']+'.avi')
        out = cv2.VideoWriter(args['save']+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30,(int(cap.get(3)),int(cap.get(4))))
	
    while(cap.isOpened):
        _ , frame = cap.read()
        output_frame = Eyeliner(frame)

        if args['save']:
            out.write(output_frame)

        cv2.imshow("Artificial Eyeliner", cv2.resize(output_frame, (600,600)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if args['save']:
        out.release()

    cap.release()
    cv2.destroyAllWindows()


def image(source):
    if os.path.isfile(source):
        img = cv2.imread(source)
        output_frame = Eyeliner(img)
        cv2.imshow("Artificial Eyeliner", cv2.resize(output_frame, (600, 600)))
        if args['save']:
            if os.path.isfile(args['save']+'.png'):
                os.remove(args['save']+'.png')
            cv2.imwrite(args['save']+'.png', output_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("File not found :( ")


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False, help="Path to video file")
    ap.add_argument("-i", "--image", required=False, help="Path to image")
    ap.add_argument("-d", "--dat", required=False, help="Path to shape_predictor_68_face_landmarks.dat")
    ap.add_argument("-t", "--thickness", required=False, help="Enter int value of thickness (recommended 0-5)")
    ap.add_argument("-c", "--color", required=False, help='Enter R G B color value', nargs=3)
    ap.add_argument("-s", "--save", required=False, help='Enter the file name to save')
    args = vars(ap.parse_args())

    if args['dat']:
        dataFile = args['dat']

    else:
        dataFile = "shape_predictor_68_face_landmarks.dat"

    color = (0,0,0)
    thickness = 2
    face_detector = dlib.get_frontal_face_detector()
    lndMrkDetector = dlib.shape_predictor(dataFile)

    if args['color']:
        color = list(map(int, args['color']))
        color = tuple(color)

    if args['thickness']:
        thickness = int(args['thickness'])

    if args['image']:
        image(args['image'])

    if args['video'] and args['video']!='webcam':
        if os.path.isfile(args['video']):
            video(args['video'])

        else:
            print("File not found :( ")

    elif args['video']=='webcam':
        video(0)
