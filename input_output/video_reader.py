
import cv2
import queue
import threading
import time


class VideoReader:
    """
    This class is used along with the VideoReaderQueue class to read the frames of the videos into a queue to speed up this process
    """

    def __init__(self, filename):
        self.filename = filename
        self.nframes = None
        self.framerate = None
        self.img_shape = None
        self._set_video_info()

    def _set_video_info(self):

        vid = cv2.VideoCapture(self.filename)
        _, img = vid.read()

        self.nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.framerate = vid.get(cv2.CAP_PROP_FPS)
        self.img_shape = img.shape

        vid.release()

    def load_video(self, interval=1):
        """
        Function Goal: Loads the images of the video

        interval: integer - Interval between frames returned. eg. 1 = every frame, 20 = every 20th frame.

        Return: a generator with the images, and the corresponding frame numbers.
        """

        def read_frames():
            vid = cv2.VideoCapture(self.filename)

            for i in frame_nums:
                vid.set(cv2.CAP_PROP_POS_FRAMES, i)

                has_frame, img = vid.read()

                if has_frame:
                    yield img
                else:
                    break

            vid.release()

        frame_nums = range(0, self.nframes, interval)

        return read_frames(), frame_nums

    def get_frame(self, n):
        """
        Return: the image at a specific frame number
        """

        vid = cv2.VideoCapture(self.filename)
        vid.set(cv2.CAP_PROP_POS_FRAMES, n)

        _, img = vid.read()

        vid.release()

        return img


class VideoReaderQueue(VideoReader):
    """
    This class is used along with the VideoReader class to read the frames of the videos into a queue to speed up this process
    """

    def __init__(self, filename, queue_size=32):

        super().__init__(filename)
        self.queue_size = queue_size

    def load_video(self, interval=1):
        """
        Function Goal: Loads the images of the video - Loads the video on a separate thread lazily.

        interval: integer - Interval between frames returned. eg. 1 = every frame, 20 = every 20th frame.

        Return: a generator with the images, and the corresponding frame numbers.
        """

        def read_frames():

            vid = cv2.VideoCapture(self.filename)

            i = 0
            while vid.isOpened():
                grabbed = vid.grab()

                if grabbed:
                    if i in frame_nums:
                        _, img = vid.retrieve()
                        q.put(img)
                else:
                    break

                i += 1

            q.put(None)  # Mark end of video
            vid.release()

        def yield_frames():

            while True:
                if q.qsize() > 0:
                    item = q.get()

                    if item is None:
                        break
                    else:

                        yield item
                else:
                    time.sleep(0.01)

        frame_nums = range(0, self.nframes, interval)

        q = queue.Queue(maxsize=self.queue_size)
        thread = threading.Thread(target=read_frames)
        thread.daemon = True
        thread.start()

        return yield_frames(), frame_nums
