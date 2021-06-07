import os
import cv2
import numpy as np
import tensorflow as tf

from .FedDataBase import FedData, shuffle


class intel(FedData):
    def load_data(self):
        data_path = os.path.join(os.path.dirname(self.local_path), 'data', 'intel_image_classification')

        # By default, we load all images in seg_train as training data
        train_image_path = os.path.join(data_path, 'seg_train', 'seg_train')
        image_labels = sorted(os.listdir(train_image_path))
        x = []
        y = []
        for image_label in image_labels:
            curr_label = image_labels.index(image_label)
            for image_file in os.listdir(os.path.join(train_image_path, image_label)):
                tmp_image_data = cv2.imread(os.path.join(train_image_path, image_label, image_file))
                tmp_image_data = cv2.resize(tmp_image_data, (150, 150), interpolation=cv2.INTER_AREA)
                x.append(tmp_image_data)
                y.append(curr_label)

        """
        You may add image preprocessing steps here.
        """

        # Formatting the data into np.array
        x = np.array(x).astype(np.float32)
        y = np.expand_dims(np.array(y).astype(np.int32), -1)
        y = tf.keras.utils.to_categorical(y, self.num_class)

        # Shuffle the data
        x, y = shuffle(x, y)

        # Set self.num_class (Required)
        # Currently we only support classification tasks, because the simulation 
        # of non-IID data needs the class labels. The non-classification tasks will
        # be supported in the future.
        self.num_class = y.shape[-1]
        
        print(x.shape, y.shape)

        return x, y