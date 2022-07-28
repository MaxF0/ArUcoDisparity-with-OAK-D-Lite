
#Basis for mono ouptut: https://docs.luxonis.com/projects/api/en/latest/samples/MonoCamera/mono_preview/
import cv2
import depthai as dai
import time
import numpy as np
from cv2 import aruco
import math



#ArUco declarations
# mtx,dist = c alibration matrix and distortion coefficients of camera calibration (https://docs.luxonis.com/en/latest/pages/calibration/)
#mtx = np.load('D:\TUBcloud\Bachelorarbeit\Code\TutorialDownloads\ArUco-marker-detection-with-DepthAi-main\datacalib_mtx_THE400.pkl', allow_pickle=True)
#dist = np.load('D:\TUBcloud\Bachelorarbeit\Code\TutorialDownloads\ArUco-marker-detection-with-DepthAi-main\datacalib_dist_THE400.pkl', allow_pickle=True)
size_of_marker = 0.052  # 52mm Breite
length_of_axis = 0.05
x_mid_Right = 0
y_mid_Right = 0
x_mid_Left = 0
y_mid_Left = 0


# Load ArUco Dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters_create()

# Create pipelinqe
pipeline = dai.Pipeline()

# Define sources and outputs
monoRight = pipeline.create(dai.node.MonoCamera)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName('right')

monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName('left')

# Properties
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)


# Linking
monoRight.out.link(xoutRight.input)
monoLeft.out.link(xoutLeft.input)




# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    #Declarations for fps counter
    startTime = time.monotonic()
    counter = 0
    fps = 0

    # Output queues will be used to get the grayscale frames from the outputs defined above
    qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)

    # Calc for stereo Depth
    calibData = device.readCalibration()
    lr_extrinsics = np.array(calibData.getCameraExtrinsics(
    dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT, useSpecTranslation=True))
    translation_vector = lr_extrinsics[0:3, 3:4].flatten()
    baseline_cm = math.sqrt(np.sum(translation_vector ** 2))
    # to get in mm baseline (7,5 cm)
    baseline = baseline_cm * 10

    hfov = calibData.getFov(dai.CameraBoardSocket.LEFT)  # (72,9)
    print(hfov)
    pixel_width = 640  # for resolution 640x 400 (THE_400_P)
    mono_width = 640
    mono_heigth = 400
    focal_length_pixels = pixel_width * 0.5 / math.tan(hfov * 0.5 * math.pi/180)

    while True:
        #FPS counter
        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        
        # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
        inRight = qRight.tryGet()
        inLeft = qLeft.tryGet()

        disparity = None
        if inRight and inLeft is not None:
            # ArUco on rigth side
            frameRight = inRight.getCvFrame()
            # ArUco processing (ArUco only uses rigth camera) (on host)  
            cornersRight, idsRight, rejectedImgPointsRight = aruco.detectMarkers(
                frameRight, aruco_dict, parameters=parameters)
            # Draw Detected Markers: frameRight->frame_markers
            frame_markersRight = aruco.drawDetectedMarkers(
                frameRight.copy(), cornersRight, idsRight)
            for cornerRight in cornersRight:
                x_mid_Right = (cornerRight[0][1][0]+cornerRight[0][3][0])/2
                y_mid_Right = (cornerRight[0][1][1]+cornerRight[0][3][1])/2

            # Display fps
            cv2.putText(frame_markersRight, "Fps: {:.2f}".format(
                fps), (2, 396), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            #PutText
            cv2.putText(frame_markersRight, 'X_Pixel: '+str(x_mid_Right), (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            cv2.putText(frame_markersRight, 'Y_Pixel: '+str(y_mid_Right), (200, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            cv2.imshow('ArucoRight', frame_markersRight)

            #Aruco on left side
            frameLeft = inLeft.getCvFrame()
            # ArUco processing (ArUco only uses rigth camera) (on host)
            cornersLeft, idsLeft, rejectedImgPointsLeft = aruco.detectMarkers(
                frameLeft, aruco_dict, parameters=parameters)
            # Draw Detected Markers: frameRight->frame_markers
            frame_markersLeft = aruco.drawDetectedMarkers(
                frameLeft.copy(), cornersLeft, idsLeft)
            for cornerLeft in cornersLeft:
                x_mid_Left = (cornerLeft[0][1][0]+cornerLeft[0][3][0])/2
                y_mid_Left = (cornerLeft[0][1][1]+cornerLeft[0][3][1])/2

            # Stereo Interference: 
            # src: https://github.com/luxonis/depthai-experiments/tree/master/gen2-triangulation
            # and: https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/
            x_delta = x_mid_Left-x_mid_Right
            y_delta = y_mid_Left-y_mid_Right
            disparity_pixels = math.sqrt(x_delta ** 2 + y_delta ** 2)            

            #Calc Depth
            if disparity_pixels == 0:
                depth = 9999999999999  # To avoid dividing by 0 error
            else:
                depth = focal_length_pixels * baseline / disparity_pixels

            # Calc Position
            x = x_mid_Right
            y = y_mid_Right

            bb_x_pos = x - mono_width / 2
            bb_y_pos = y - mono_heigth / 2

            angle_x = math.atan(math.tan(hfov / 2.0) *bb_x_pos / (640 / 2.0))
            angle_y = math.atan(math.tan(hfov / 2.0) *bb_y_pos / (640 / 2.0))

            z = depth
            x = z * math.tan(angle_x)
            y = -z * math.tan(angle_y)
            #if (depth != 9999999999999):print(f"x {x}, y {y}, z {z}")


            # Display
            # Display fps
            cv2.putText(frame_markersLeft, "Fps: {:.2f}".format(
                fps), (2, 396), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            #PutText
            cv2.putText(frame_markersLeft, 'X_Pixel: '+str(x_mid_Left), (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            cv2.putText(frame_markersLeft, 'Y_Pixel: '+str(y_mid_Left), (200, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            cv2.imshow('ArucoLeft', frame_markersLeft)
            
            combined = cv2.addWeighted(
                frameRight, 0.5, frameLeft, 0.5, 0)
            cv2.putText(combined, 'disparity'+str(disparity_pixels), (10, 90),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            cv2.putText(combined, f"x {x}", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            cv2.putText(combined, f"y {y}", (10, 50),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            cv2.putText(combined, f"z {z}", (10, 70),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            cv2.imshow('combined', combined)

        if cv2.waitKey(1) == ord('q'):
            break
