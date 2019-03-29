from segmentation_models import Unet
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import to_categorical
#from segmentation_models.backbones import get_preprocessing
import numpy as np
#model = Unet()
model = Unet('resnext50',encoder_weights=None,input_shape = (None,None,3),classes = 10)
#model = Unet('resnet34', classes=3, activation='softmax')
#model = Unet('resnet34', input_shape=(None, None, 6), encoder_weights=None)
#preprocess_input = get_preprocessing('resnext50')
SMOOTH = 1e-12
def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    intersection = K.sum(gt * pr, axis=axes)
    union = K.sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = K.mean(iou, axis=0)

    # weighted mean per class
    iou = K.mean(iou * class_weights)

    return iou


test_images = []
labels = []
for i in range(100):
    test_images.append(np.random.rand(128,128,3))
    rand = np.random.randint(10,size = (128,128))
    labels.append(rand)
test_images = np.array(test_images)
labels = np.array(labels)
x_train, x_val, y_train, y_val = train_test_split(test_images, labels, test_size=0.33, random_state=42)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
print(x_train.shape)
print(y_train.shape)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[iou_score])
model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=100,
    validation_data=(x_val, y_val),
)