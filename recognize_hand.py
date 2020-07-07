import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


def convert(image):
    img = cv2.imread(image)
    img_resize = cv2.resize(img, (32, 32))
    return np.float32(img_resize)

class Recognition:

    def threshold_image(self, img_frame):
        blurred_frame = cv2.medianBlur(img_frame, 5)
        # cv2.imshow("bll", blurred_frame)
        # ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # plt.imshow(ycrcb_img[:,:,0])
        x, y = blurred_frame[:, :, 0].shape
        #print("Width and Height of the Image is - ({},{})".format(y, x))
        img_res = cv2.resize(blurred_frame, (y // 4, x // 4))
        x_r, y_r = img_res[:, :, 0].shape
        img_hsv = cv2.cvtColor(img_res, cv2.COLOR_BGR2HSV)
        #print("Width and Height of the Image is - ({},{})".format(y_r, x_r))
        copy = img_hsv[0:y_r // 2 + 10, 0:x_r // 2 + 10].copy()
        #cv2.imshow("cop", copy)
        cv2.imwrite("hsv.png", copy)
        # Skin color upper and lower bound in HSV Map
        # pixel_colors = copy.reshape((np.shape(copy)[0]*np.shape(copy)[1], 3))
        # norm = colors.Normalize(vmin=-1.,vmax=1.)
        # norm.autoscale(pixel_colors)
        # pixel_colors = norm(pixel_colors).tolist()
        # h, s, v = cv2.split(copy)
        # fig = plt.figure()
        # axis = fig.add_subplot(1, 1, 1, projection="3d")
        # axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
        # axis.set_xlabel("Hue")
        # axis.set_ylabel("Saturation")
        # axis.set_zlabel("Value")
        # plt.show()

        # Every color except white
        low = np.array([0, 50, 0])
        high = np.array([15, 255, 255])
        mask = cv2.inRange(copy, low, high)
        result = cv2.bitwise_and(copy, copy, mask=mask)
        #cv2.imshow("res", result)

        res, thresh = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY)
        #print(thresh.shape)
        return img_res, thresh
        #plt.imshow(thresh, cmap="gray")

    def draw_contours(self, org_frame, thresh_image):
        thresh_copy = thresh_image.copy()
        # since findContours alters the image
        _, contours, _ = cv2.findContours(thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1
        max_index =0
        #cv2.imshow('Canny Edges After Contouring', thresh_copy)

        #print("Number of Contours found = " + str(len(contours)))

        # Draw all contours
        # -1 signifies drawing all contours
        if len(contours) > 4:
            for i, cnt in enumerate(contours):
                cnt_temp = contours[i]
                area = cv2.contourArea(cnt_temp)
                if area > max_area:
                    max_area = area
                    max_index = i

            cnt = contours[max_index]
            x, y, w, h = cv2.boundingRect(cnt)
            bounded_img = cv2.rectangle(org_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return bounded_img
        else:
            return None
        #plt.imshow(cv2.rectangle(img_res, (x, y), (x + w, y + h), (0, 255, 0), 2))


#### Main #######
recog = Recognition()
from keras.models import load_model
import numpy as np
import cv2
import pyautogui

model = load_model("cl_models/20191103230955model.h5")

classes = {"background": 0, "l": 1, "thumbsup": 2, "up": 3}

cap = cv2.VideoCapture(0)
index = 0
bg_captured = True
while(True):
    import datetime

    # YYYYMMDDHHMMSS
    date_time = str(datetime.datetime.utcnow()).replace("-", "").replace(" ", "").replace(":", "").split(".")[0]
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow("frame", frame)

    if bg_captured:
        #Remove background from the original mask
        # Capture the background
        #backSub = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=50, detectShadows=True)
        #bg_captured = True
        #fgmask = backSub.apply(frame)
        #kernel = np.ones((3, 3), np.uint8)
        #fgmask = cv2.erode(fgmask, kernel, iterations=1)
        #res = cv2.bitwise_and(frame, frame, mask=fgmask)
        #cv2.imshow("fg", fgmask)
        img_resized, thresholded_image = recog.threshold_image(frame)
        #print(thresholded_image.shape)
        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (0, 70)

        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (90, 0)

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv2.rectangle(thresholded_image, start_point, end_point, color, thickness)
        cv2.imshow("th", image)
        #img_pred = np.float32(cv2.resize(thresholded_image, (32, 32)))
        '''
        pred = model.predict(img_pred.reshape(1, 32, 32, 1)).reshape(-1)
        for i, val in enumerate(pred):
            if val >= .9999999999976:
                pred_class_name = [key for key, val in classes.items() if val == i]
                if "background" not in pred_class_name:
                    if "l" in pred_class_name:
                        print("Prediction is - {} with confidence of {}".format(pred_class_name, val * 100))
                        pyautogui.hotkey("volumedown")
                    elif "up" in pred_class_name:
                        print("Prediction is - {} with confidence of {}".format(pred_class_name, val * 100))
                        pyautogui.hotkey("volumeup")
                    else:
                        print("Prediction is - {} with confidence of {}".format(pred_class_name, val * 100))
                        pyautogui.screenshot("screenshot/screen_" + date_time + ".png")
            else:
                pass
        '''

    #if cv2.waitKey(10) == ord("b"):
        # Capture the background
     #   backSub = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=50, detectShadows=True)
      #  bg_captured = True

    if cv2.waitKey(1) == ord("p"):
        img_pred = np.float32(cv2.resize(thresholded_image, (32, 32)))
        pred = model.predict(img_pred.reshape(1, 32, 32, 3)).reshape(-1)
        for i, val in enumerate(pred):
		# Highest of the prediction value being confident.
            if val >= .9999999999976:
                pred_class_name = [key for key, val in classes.items() if val == i]
                if "background" not in pred_class_name:
                    if "l" in pred_class_name:
                        print("Prediction is - {} with confidence of {}".format(pred_class_name, val * 100))
                        pyautogui.hotkey("volumedown")
                    elif "up" in pred_class_name:
                        print("Prediction is - {} with confidence of {}".format(pred_class_name, val * 100))
                        pyautogui.hotkey("volumeup")
                    else:
                        #print("Prediction is - {} with confidence of {}".format(pred_class_name, val * 100))
                        #pyautogui.screenshot("screenshot/screen_" + date_time + ".png")
                        pyautogui.hotkey("mute")
            else:
                    pass


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
