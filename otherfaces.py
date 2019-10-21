# -*- codeing: utf-8 -*-
import sys
import os
import cv2
import dlib

input_dir = './input_img'
output_dir = './other_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Use the frontal_face_detector that comes with dlib as our feature extractor
detector = dlib.get_frontal_face_detector()

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path+'/'+filename
            # read photos from files
            img = cv2.imread(img_path)
            # convert to gray images
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # use the detector function to get features, dets are the return results
            dets = detector(gray_img, 1)

            # use enumerate function to get the list element and their subscript
            # subscript i is the face image's number
            #left：The distance from the left side of the face to the left edge of the image；
            #right：The distance from the right side of the face to the left edge of the image
            #top：The distance from the top side of the face to the left edge of the image；
            #bottom：The distance from the bottom side of the face to the left edge of the image
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                # img[y:y+h,x:x+w]
                face = img[x1:y1,x2:y2]
                # adjust the size of the images
                face = cv2.resize(face, (size,size))
                cv2.imshow('image',face)
                # save images
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)
