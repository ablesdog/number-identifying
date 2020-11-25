# ------------------------------------------------------------------------------------

import numpy as np
import imutils
import mnist
import cv2
import os

# ----------------------------------------------------------------------------------------------------------------------


def measure(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erode = cv2.dilate(thresh, kernel)
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for i, contour in enumerate(contours):

        x, y, w, h = cv2.boundingRect(contour)
        if h > 10:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            hcenter = x + w / 2
            vcenter = y + h / 2
            if h > w:
                w = h
                x = hcenter - (w / 2)
            else:
                h = w
                y = vcenter - (h / 2)

            result = out[int(y):int(y + h), int(x):int(x + w)]

            black = [0, 0, 0]
            constant = cv2.copyMakeBorder(result, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=black)
            cv2.imwrite('./out/out.png', constant)

            predict = mnist.recognition('./out/out.png')
            results.append([predict, x])

            text = "{}".format(predict[0])
            cv2.putText(image, text, (np.int(x), np.int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('rect', image)
            cv2.waitKey(0)

    results = sorted(results, key=(lambda x: x[1]))
    out = ''
    for result in results:
        out += str(result[0][0])

    print(out)


# ----------------------------------------------------------------------------------------------------------------------


files = os.listdir('./data/')
for file in files:
    if file.endswith('jpg') or file.endswith('png'):
        image_path = './data/' + file

        image = cv2.imread(image_path)
        image = imutils.resize(image, width=600)
        measure(image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------

















