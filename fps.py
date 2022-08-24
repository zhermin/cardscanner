"""Simple FPS counter for webcam feed without any processing"""
import cv2
import time
from Camera import Camera

# Initialise FPS timings
prev_frame_time, cur_frame_time = 0, 0

camera = Camera(1)
runtimes = []

while True:
    # Get frame from camera
    frame_got, frame = camera.get_frame()
    if not frame_got:
        print("Failed to get frame")
        break

    # Calculate FPS
    cur_frame_time = time.time()
    if prev_frame_time != 0:
        fps = 1 / (cur_frame_time - prev_frame_time)
    else:
        fps = 0

    runtimes.append(fps)
    prev_frame_time = cur_frame_time

    # Display FPS
    cv2.putText(
        frame,
        "FPS: {:.2f}".format(fps),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # Display frame
    cv2.imshow("Frame", frame)

    # Exit if ESC is pressed
    if cv2.waitKey(1) == 27:
        break

camera.release()

print("Average FPS: {:.2f}".format(sum(runtimes) / len(runtimes)))
