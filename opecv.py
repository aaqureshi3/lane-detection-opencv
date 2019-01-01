import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intecept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1) #array containing slope and y intercept of line
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        if len(left_fit) != 0 and len(right_fit) != 0:
            left_line = make_coordinates(image, left_fit_average)
            right_line = make_coordinates(image, right_fit_average)
            return np.array([left_line, right_line])
        elif len(left_fit) != 0:
            left_line = make_coordinates(image, left_fit_average)
            return np.array([left_line])
        elif len(right_fit) != 0:
            right_line = make_coordinates(image, right_fit_average)
            return np.array([right_line])
        else:
            return np.array([[0,0,0,0]])
    else:
        return np.array([[0,0,0,0]])

def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)
    canny = cv.Canny(blur,50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(line_image, (x1,y1),(x2,y2),(255,0,0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    #[(100,height), (900, height), (490,290)]
    polygons = np.array([
        [(100,height), (1000, height), (550,250)]
        ])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, polygons, 255)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

# image = cv.imread("lane.PNG")
# plt.imshow(image)
# plt.show()

cap = cv.VideoCapture("solidWhiteRight.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv.HoughLinesP(cropped_image, 2, np.pi/180, 150, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intecept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    cv.imshow("result", combo_image)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()