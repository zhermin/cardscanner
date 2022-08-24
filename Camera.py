import numpy as np
import cv2


class Camera:
    def __init__(self, video_src: int) -> None:
        self.video_src = video_src
        if self.video_src is None:
            print("Video source not assigned, default webcam will be used")
            self.video_src = 0
        self.cap = cv2.VideoCapture(self.video_src)

    def get_frame_size(self) -> tuple[int]:
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def get_frame(self) -> tuple[bool, np.ndarray]:
        frame_got, frame = self.cap.read()
        # If the frame comes from webcam, flip it so it looks like a mirror
        if isinstance(self.video_src, int):
            frame = cv2.flip(frame, 2)
        return frame_got, frame

    def release(self) -> None:
        self.cap.release()
