import cv2


class VideoReader:
    """
    Efficient video reader that supports reading individual frames or loading the entire video.
    Keeps the video file open for efficient frame reading without caching old frames.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        # open the video
        self.vid = cv2.VideoCapture(self.file_path)
        if not self.vid.isOpened():
            raise ValueError(f"Unable to open video file: {self.file_path}")
        # get video properties
        self.nframes = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = self.vid.get(cv2.CAP_PROP_FPS)
        # get video shape by reading first frame
        _, img = self.vid.read()
        self.img_shape = img.shape if img is not None else None
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_frame(self, frame_number):
        """
        Retrieve the image at a specific frame number.
        Keeps the video file open to avoid overhead of reopening the file.
        """
        # check the video has this frame
        if frame_number > self.nframes:
            raise ValueError(
                f"Cannot get frame number '{frame_number}'. Video at '{self.file_path} only has '{self.nframes}'."
            )
        # re-open the video if closed accidentally
        if not self.vid.isOpened():
            self.vid = cv2.VideoCapture(self.file_path)
        # get the frame
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, img = self.vid.read()
        if not success:
            raise ValueError(f"Frame '{frame_number}' could not be read from video at '{self.file_path}'")

        return img

    def load_video(self, interval=1):
        """
        Generator that loads the entire video, frame by frame, with the specified interval.
        Does not cache frames to conserve memory.
        """
        frame_nums = range(0, self.nframes, interval)
        for i in frame_nums:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, img = self.vid.read()
            if success:
                yield img
            else:
                break

    def reset(self):
        """
        Resets the video to the beginning, useful for reprocessing or re-reading frames.
        """
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        """
        Closes the video file when done. Call this when no longer reading frames.
        """
        if self.vid.isOpened():
            self.vid.release()
