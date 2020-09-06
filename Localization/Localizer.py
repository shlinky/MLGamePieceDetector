import os
import sys
import numpy as np
from six import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import time
sys.path.append('C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Classification/')
from Training.Model_Builder import ModelBuilder

class Localizer:
  def __init__(self, model, detection_weights = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model'):
    self.class_model = model
    self.detection_weights = detection_weights
    self.load_detector()

  def load_image_into_numpy_array(self, path):
    #Load an image from file into a numpy array.
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def load_detector(self):
    start_time = time.time()
    tf.keras.backend.clear_session()
    self.detect_model = tf.saved_model.load(self.detection_weights)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Loading time: ' + str(elapsed_time) + 's')

  def get_imgs_from_bb(self, img, bb, height, width):
    img_shape = img.shape
    imgs = []
    nbb = []
    for b in bb:
      i = [0 for n in range(4)]
      i[0] = int(b[0] * img_shape[0])
      i[1] = int(b[1] * img_shape[1])
      i[2] = int(b[2] * img_shape[0])
      i[3] = int(b[3] * img_shape[1])
      box = img[i[0]:i[2], i[1]:i[3]]
      if ((box.shape[0] < 10) or (box.shape[1] < 10)) or (abs(1 - (box.shape[0]/box.shape[1])) > 0.16):
        continue

      im = Image.fromarray(box)
      im_resized = im.resize((width, height))
      box = np.array(im_resized)

      imgs.append(box)
      nbb.append(b)
    imgs = np.stack(imgs)

    print('boxes shape: ', imgs.shape)
    return (imgs, nbb)

  def get_bounding_boxes(self, img, conf_threshold = 0.7):
    start_time = time.time()
    detections = self.detect_model(img)
    end_time = time.time()
    print("bounding box time: ", end_time - start_time)

    bnd_boxes = detections['detection_boxes'][0].numpy().tolist()
    num_boxes = 0
    for n, i in enumerate(detections['detection_scores'][0]):
      if i < conf_threshold:
        num_boxes = n
        break
    while num_boxes < len(bnd_boxes):
      bnd_boxes.pop(num_boxes)

    return bnd_boxes
  
  #gives back classification inferences and relative bounding box coordinates
  #bounding box format: (y1, x1, y2, x2)
  def call(self, img_path, display = False):
    image_np = self.load_image_into_numpy_array(img_path)
    img_shape = image_np.shape
    input_tensor = np.expand_dims(image_np, 0)
    bb = self.get_bounding_boxes(input_tensor)
    boxes, bb = self.get_imgs_from_bb(image_np, bb, 180, 180)

    print("bounding box coords: ", bb)
    print("image shape: ", img_shape)

    class_pred = self.class_model.predict(boxes)
    print(class_pred)

    if display:
      plt.title("Original Image")
      plt.imshow(image_np)
      plt.show()
      for n, i in enumerate(boxes):
        if class_pred[n] > 0.6:
          plt.title("Game Piece")
          plt.imshow(i)
          plt.show()

    return {"predictions": class_pred, "coords": bb}

if __name__ == '__main__':
  image_dir = 'C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Classification/Data/extra_imgs/1data39.png'
  mb = ModelBuilder(ckpt = 'C:/Users/shloks/Documents/robproj/MLGamePieceDetector/Classification/Training/checkpoints/LL_NEWSA')
  l = Localizer(mb.get_model())
  result = l.call(image_dir, display = True)