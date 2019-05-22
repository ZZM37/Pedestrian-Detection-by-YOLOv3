#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import os
import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from PIL import Image, ImageFont, ImageDraw
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from YOLOv3.model import yolo_eval, tiny_yolo_body, create_tiny_model
from YOLOv3.utils import get_classes, get_anchors, bbox_iou, letterbox_image, data_generator_wrapper

class YOLO(object):
    def __init__(self):
        self.model_path = 'logs\\INRIA_trained_weights_network.h5'    # model path or trained weights path 
        # 'logs/000/ep013-loss6.261-val_loss6.037.h5'
        self.log_dir = 'logs/'
        self.anchors_path = 'model_data/INRIA_anchor_tiny.txt'  #'New_anchors.txt'
        self.classes_path = 'model_data/classes.txt'   #'model_data/coco_classes.txt'   
        self.annotation_path_train = 'model_data/INRIA_train.txt'
        self.annotation_path_val = 'model_data/INRIA_val.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def train(self):
        annotation_path = self.annotation_path_train
        annotation_path2 = self.annotation_path_val
        log_dir = self.log_dir
        classes_path = self.classes_path
        anchors_path = self.anchors_path
        class_names = self.class_names
        num_classes = len(class_names)
        anchors = self.anchors

        input_shape = self.model_image_size # multiple of 32, hw

        model = create_tiny_model(input_shape, anchors, num_classes,
            weights_path='model_data/yolo_weights_tiny.h5', 
            freeze_body=1)
                # weights_path='model_data/yolo_weights_tiny_person.h5'
        # make sure you know what you freeze

        logging = TensorBoard(log_dir=log_dir)
        checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
            monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        with open(annotation_path) as f:
            lines = f.readlines()
        with open(annotation_path2) as f2:
            lines2 = f2.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.shuffle(lines2)
        np.random.seed(None)
        num_train = int(len(lines))
        num_val = int(len(lines2))

        if True:
            for i in range(len(model.layers)):
                model.layers[i].trainable = True
            model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
            print('Unfreeze all of the layers.')

            batch_size = 16 # note that more GPU memory is required after unfreezing the body
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines2[:num_val], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=120,
                initial_epoch=0,
                callbacks=[checkpoint, early_stopping, logging])    # delete logging
            model.save_weights(log_dir + 'INRIA_trained_weights.h5')

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6 # default setting   

        try:
            self.yolo_model = load_model(model_path, compile=False)
            print('load model!')
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)

            #plot_model(self.yolo_model, to_file='yolov3_tiny_person.png',show_shapes=True,show_layer_names=True)

            self.yolo_model.load_weights(self.model_path, by_name=True, skip_mismatch=True) # make sure model, anchors and classes match
            print('load weights!')
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 4), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, output_path = "", name="", is_evaluate = False):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                    size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(2):    #for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)

        if is_evaluate:
            return out_boxes, out_scores, out_classes

        if output_path != "":
            image = np.array(image)
            cv2.imwrite(output_path+name, image)

        return image

    def close_session(self):
        self.sess.close()

    def test_img(self, img, output_path = ""):
        while True:
            #img = input('Input image filename:')
            try: 
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                break
            else:
                name = img.split('/')[-1]
                r_image = self.detect_image(image, name=name, output_path=output_path)
                r_image = np.asarray(r_image)
                cv2.imshow("result", r_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.close_session()

    def evaluate(self, path, iou_threshold):

        with open(path) as f:
            lines = f.readlines()

        p, ngt = 0, 0
        tp, tn = 0, 0
        fp, fn = 0, 0
        average_iou = 0

        for i in range(288):
            line = lines[i].split()
            image = Image.open(line[0])
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
            ngt += box.shape[0]

            out_boxes, _, _ = self.detect_image(image, is_evaluate=True)
            out_boxes = out_boxes[:, [1,0,3,2]]
            p += out_boxes.shape[0]
            #print(i, box.shape[0], out_boxes.shape[0])

            detected = []
            for e in range(out_boxes.shape[0]):
                iou = bbox_iou(out_boxes[e], box)
                m = np.argmax(iou)
                if iou[0, m] > iou_threshold and m not in detected:
                    tp += 1
                    average_iou += iou[0, m]
                    detected.append(m)

        recall = [tp/ngt*100]
        average_iou = [average_iou/tp if tp != 0 else 0]
        print(recall, average_iou)
        yolo.close_session()

if __name__ == '__main__':
    #detect_img(YOLO())
    #detect_video(YOLO(), video_path = 'C:/Users/ZZM/Desktop/WIN_20190223_15_09_40_Pro.mp4')
    
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    import cv2

    yolo = YOLO()

    #yolo.test_img(img = 'D:/data/Material/DataSets/INRIADATA/original_images/test/pos/crop001593.png', 
        #output_path = "result\\images")

    #yolo.train()
    path = "model_data/INRIA_val.txt"
    yolo.evaluate(path, iou_threshold=0.5)
