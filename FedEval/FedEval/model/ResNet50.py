import tensorflow as tf
from .BaseModel import KerasModel


class ResNet50(KerasModel):

    def __init__(self,
                 inputs_shape,
                 targets_shape,
                 lr=1e-4,
                 include_top=False,
                 optimizer='adam',
                 weights='imagenet',
                 dense_units=(512, 512,),
                 activation='relu',
                 code_version='MobileNet',
                 model_dir='log',
                 gpu_device='0'):

        self.activation = activation
        self.include_top = include_top
        self.optimizer = optimizer
        self.weights = weights
        self.dense_units = dense_units
        self.lr = lr

        super(ResNet50, self).__init__(inputs_shape=inputs_shape, targets_shape=targets_shape,
                                       code_version=code_version, model_dir=model_dir,
                                       gpu_device=gpu_device)

    def build(self):
        input_image = tf.keras.Input(shape=self.inputs_shape['x'])

        base_model = tf.keras.applications.ResNet50(
            input_shape=self.inputs_shape['x'],
            include_top=False,
            weights=self.weights,
            input_tensor=None,
            pooling=None,
            classes=self.targets_shape['y'][-1]
        )

        feature = base_model.call(input_image)
        feature = tf.keras.layers.Flatten()(feature)

        for unit in self.dense_units:
            feature = tf.keras.layers.Dense(unit, activation=self.activation)(feature)
            feature = tf.keras.layers.Dropout(0.2)(feature)

        prediction = tf.keras.layers.Dense(self.targets_shape['y'][-1], activation='softmax')(feature)

        self.model = tf.keras.models.Model(inputs=input_image, outputs=prediction)

        if self.optimizer.lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        else:
            optimizer = tf.keras.optimizers.SGD(lr=self.lr)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        super(ResNet50, self).build()