
# coding: utf-8

# In[5]:

import h5py
import numpy as np
from sklearn.utils import shuffle
import gc
np.random.seed(2017)

X_train = []
X_test = []

for filename in ["gap_ResNet50.h5", "gap_Xception.h5", "gap_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)


# In[6]:

from keras.models import *
from keras.layers import *

input_tensor = Input(X_train.shape[1:])
x = input_tensor
x = Dropout(0.5)(x)
# x = Dense(1, activation='sigmoid')(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# In[8]:

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# In[4]:

model.fit(X_train, y_train, batch_size=128, nb_epoch=20, validation_split=0.2)


# In[5]:

model.save('model.h5')


# In[6]:

y_pred = model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)


# In[7]:

import pandas as pd
from keras.preprocessing.image import *

df = pd.read_csv("sample_submission.csv")

image_size = (224, 224)
gen = ImageDataGenerator()
# test_generator = gen.flow_from_directory("test2", image_size, shuffle=False,
#                                          batch_size=16, class_mode=None)
test_generator = gen.flow_from_directory("test-small-dataset", image_size, shuffle=False,
                                         batch_size=16, class_mode=None)

test_generator.filenames.sort()
for i, fname in enumerate(test_generator.filenames):
    index = fname[fname.rfind('/')+1:fname.rfind('.')]
    print("image %s:\t%f" % (fname, y_pred[i]))

# for i, fname in enumerate(test_generator.filenames):
#     index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
#     df.set_value(index-1, 'label', y_pred[i])
#     if index < 100:
#         print("image %s:\t%f" % (fname, y_pred[i]))
df.to_csv('pred.csv', index=None)
df.head(10)


# In[ ]:

gc.collect()

