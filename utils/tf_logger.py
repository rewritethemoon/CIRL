import tensorflow as tf
import numpy as np
import io
from PIL import Image

class TFLogger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        with self.writer.as_default():
            for i, img in enumerate(images):
                img = np.array(img)
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                tf.summary.image(f"{tag}/{i}", np.expand_dims(img, 0), step=step)
            self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
            self.writer.flush()
