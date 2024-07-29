import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense, Dropout

""" 
yolo_architecture:
Tuple is (kernel_size, filters, stride, padding) 
"M" is maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and int with number of repeats
"""

yolo_architecture = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

#out_channels is the number of kernels being applied

#kwargs allows me to not list all parameters in function header and add them in the function if needed

#super(CNNBlock, self).__init__(**kwargs) is a best practice in Python, especially in the context of TensorFlow Keras,
#ensuring that your custom layer properly inherits and initializes all the necessary functionality from tf.keras.layers.Layer

#use_bias is false since batch normalization makes a bias parameter unessasary

#this function creates a CNNblock that does conv, batchnorm, and leakyRelu
class CNNBlock(Layer):
    def __init__(self, out_channels, kernel_size=3, strides=1, padding='same', **kwargs):
        super(CNNBlock, self).__init__(**kwargs)
        # initialize convultional layer
        self.conv = Conv2D(out_channels, kernel_size, strides=strides, padding=padding, use_bias=False)
        #initalize batch normalization
        self.batchnorm = BatchNormalization()
        #initialize LeakyRelu activation function which is x if > 0 and 0.1x < 0
        self.leakyrelu = LeakyReLU(alpha=0.1)

    def call(self, inputs):
        #making model conv -> batchnorm -> leakyrelu
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x


#split_size is the number of cells by cells to divide picture into
#num_boxes is the number of bounding boxes predicted per cell
#num_classes is number of classes each box can predict
#in channels = 3 because we are using rgb images

class Yolov1(Model):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20, **kwargs):
        super(Yolov1, self).__init__(**kwargs)
        self.architecture = yolo_architecture
        self.in_channels = in_channels
        #darknet is the conv layers as well as the FC layers
        self.darknet = self.create_conv_layers(yolo_architecture)
        #calling the create_fully connected layers 
        self.fcs = self.create_fcs(split_size, num_boxes, num_classes)

    def call(self, x):
        #making model to be darknet (conv layers) -> FC
        x = self.darknet(x)
        x = Flatten()(x)
        return self.fcs(x)

    def create_conv_layers(self, architecture):
        #initialize empty layers list
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            # add conv layer since tuples in yolo_architecture are conv layers
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                    x[1], kernel_size=x[0], strides=x[2], padding='valid' if x[3]==0 else 'same')
                ]

            # add Max pool layer since strings in yolo_architecture represent maxpool2d
            elif type(x) == str:
                layers += [
                    MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
                ]

            # Add a repitition of conv layers becase lists represent that in yolo_architecture
            elif type(x) == list:
                conv1 = x[0] # Tuple
                conv2 = x[1] # Tuple
                num_repeats = x[2] # Integer
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(conv1[1], kernel_size=conv1[0], strides=conv1[2], padding='valid' if conv1[3]==0 else 'same')
                    ]
                    layers += [
                        CNNBlock(conv2[1], kernel_size=conv2[0], strides=conv2[2], padding='valid' if conv2[3]==0 else 'same')
                    ]

        #Return completed conv layers for yolo model
        return tf.keras.Sequential(layers) 

    def create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return tf.keras.Sequential([
            Flatten(),
            Dense(496), #original paper should be 496
            Dropout(0.0),
            LeakyReLU(alpha=0.1),
            Dense(S * S * (C + B * 5))
        ])

# def test(S=7, B=2, C=20):
#     model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
#     x = tf.random.normal((2, 448, 448, 3))
#     print(model(x).shape)

# test()