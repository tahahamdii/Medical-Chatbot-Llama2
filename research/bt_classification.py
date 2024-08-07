# 1. IMPORTING LIBRARIES AND DATASET
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import cv2
from skimage import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K
import plotly.express as px
import random
import glob
from sklearn.preprocessing import StandardScaler, normalize
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model, save_model

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/MyDrive/lgg-mri-segmentation/Dataset/data.csv')
data.info()

age_counts = data["age_at_initial_pathologic"].value_counts()
fig = px.bar(age_counts, title="Age of patients")
fig.update_layout(
    xaxis_title="Age",
    yaxis_title="Frequency",
    title_x=0.5,
    showlegend=False
)
fig.show()

# This shows the first 10 rows of the patient data
data.head(10)

data_map = []
for sub_dir_path in glob.glob("/content/drive/MyDrive/lgg-mri-segmentation/Dataset/" + "*"):
    # if os.path.isdir(sub_path_dir):
    try:
        dir_name = sub_dir_path.split('/')[-1]
        for filename in os.listdir(sub_dir_path):
            image_path = sub_dir_path + '/' + filename
            data_map.extend([dir_name, image_path])
    except Exception as e:
        print(e)

df = pd.DataFrame({"patient_id": data_map[::2],
                   "path": data_map[1::2]})
df.head()

# Path to the images and the mask images of the Brain MRI
df_imgs = df[~df['path'].str.contains("mask")]
df_masks = df[df['path'].str.contains("mask")]

# File path line length images for later sorting
BASE_LEN = 89  #
END_IMG_LEN = 4  #
END_MASK_LEN = 9  #

# Data sorting
imgs = sorted(df_imgs["path"].values, key=lambda x: int(x[BASE_LEN:-END_IMG_LEN]))
masks = sorted(df_masks["path"].values, key=lambda x: int(x[BASE_LEN:-END_MASK_LEN]))

# Sorting check
idx = random.randint(0, len(imgs) - 1)
print("Path to the Image:", imgs[idx], "\nPath to the Mask:", masks[idx])

### 1.3 Creating the final data frame:

# Make a dataframe with the images and their corresponding masks and patient ids
# Final dataframe
brain_df = pd.DataFrame({"patient_id": df_imgs.patient_id.values,
                         "image_path": imgs,
                         "mask_path": masks
                         })


# Make a function that search for the largest pixel value in the masks, because that will indicate if the image have
# a corresponding mask with a tumor or not , also add this column to the dataframe
def pos_neg_diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0:
        return 1
    else:
        return 0


brain_df['mask'] = brain_df['mask_path'].apply(lambda x: pos_neg_diagnosis(x))
brain_df

# 2. DATA VISUALISATION"""

# How many non-tumors (0) and tumors (1) are in the data
brain_df['mask'].value_counts()

# Graphic Visualisation of the above counts as bar plots
# using plotly to create interactive plots

fig = go.Figure([go.Bar(x=brain_df['mask'].value_counts().index,
                        y=brain_df['mask'].value_counts(),
                        width=[.4, .4]
                        )
                 ])
fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=4, opacity=0.4
                  )
fig.update_layout(title_text="Mask Count Plot",
                  width=700,
                  height=550,
                  yaxis=dict(
                      title_text="Count",
                      tickmode="array",
                      titlefont=dict(size=20)
                  )
                  )
fig.update_yaxes(automargin=True)
fig.show()

# How the image of a tumor looks like and how is the same Brain MRI scan is present for the image.
for i in range(len(brain_df)):
    if cv2.imread(brain_df.mask_path[i]).max() > 0:
        break

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.imread(brain_df.mask_path[i]));
plt.title('Tumor Location')

plt.subplot(1, 2, 2)
plt.imshow(cv2.imread(brain_df.image_path[i]));

# Basic visualizations: Visualize the images (MRI and Mask) in the dataset separately

fig, axs = plt.subplots(6, 2, figsize=(16, 26))
count = 0
for x in range(6):
    i = random.randint(0, len(brain_df))  # select a random index
    axs[count][0].title.set_text("Brain MRI")  # set title
    axs[count][0].imshow(cv2.imread(brain_df.image_path[i]))  # show MRI
    axs[count][1].title.set_text("Mask - " + str(brain_df['mask'][i]))  # plot title on the mask (0 or 1)
    axs[count][1].imshow(cv2.imread(brain_df.mask_path[i]))  # Show corresponding mask
    count += 1

fig.tight_layout()

count = 0
i = 0
fig, axs = plt.subplots(12, 3, figsize=(20, 50))
for mask in brain_df['mask']:
    if (mask == 1):
        img = io.imread(brain_df.image_path[i])
        axs[count][0].title.set_text("Brain MRI")
        axs[count][0].imshow(img)

        mask = io.imread(brain_df.mask_path[i])
        axs[count][1].title.set_text("Mask")
        axs[count][1].imshow(mask, cmap='gray')

        img[mask == 255] = (255, 0, 0)  # change pixel color at the position of mask
        axs[count][2].title.set_text("MRI with Mask")
        axs[count][2].imshow(img)
        count += 1
    i += 1
    if (count == 12):
        break
fig.tight_layout()


# 3. CREATING TEST/TRAIN/VALIDATION SET"""

brain_df_train = brain_df.drop(columns=['patient_id'])
# Convert the data in mask column to string format, to use categorical mode in flow_from_dataframe
brain_df_train['mask'] = brain_df_train['mask'].apply(lambda x: str(x))
brain_df_train.info()

train, test = train_test_split(brain_df_train,test_size = 0.15)
print(train.values.shape)
print(test.values.shape)

train.head()
### 3.1 Seeing how many tumors are in the train and test set, respectively"""

# using plotly to create interactive plots


fig = go.Figure([go.Bar(x=train['mask'].value_counts().index,
                        y=train['mask'].value_counts(),
                        width=[.4, .4],
                       )
                ])
fig.update_traces(marker_color=['darkolivegreen', 'firebrick'], opacity = 0.7
                 )

fig.update_layout(title_text="Tumor Count Train Set",
                  width=700,
                  height=550,
                  yaxis=dict(
                             title_text="Count",
                             tickmode="array",
                             titlefont=dict(size=20)
                           )
                 )

fig.update_yaxes(range = list([0,3000]))
fig.update_xaxes(tick0 = 0, dtick = 1)

fig.show()


fig3 = go.Figure([go.Bar(x=test['mask'].value_counts().index,
                        y=test['mask'].value_counts(),
                        width=[.4, .4]
                       )
                ])
fig3.update_traces(marker_color=['darkolivegreen', 'firebrick'], opacity = 0.7
                 )
fig3.update_layout(title_text="Tumor Count Test Set",
                  width=700,
                  height=550,
                  yaxis=dict(
                             title_text="Count",
                             tickmode="array",
                             titlefont=dict(size=20)
                           )
                 )

fig3.update_yaxes(range = list([0,3000]))
fig3.update_xaxes(tick0 = 0, dtick = 1)

fig3.show()

# 4. CLASSIFICATION MODEL TO DETECT EXISTENCE OF TUMOR

### 4.1 Batch size


import numpy as np
import matplotlib.pyplot as plt

# Table
fig, ax = plt.subplots(1, 1)
data = [[0.883, 0.801, 0.783]]
column_labels = ['Batchsize 16', 'Batchsize 32', 'Batchsize 64']
row_label = ['Accuracy']
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data, colLabels=column_labels, rowLabels=row_label, loc="center")

plt.show()

# Barplot
# Make a random dataset:
height = [0.883, 0.801, 0.783]
bars = ('16', '32', '64')
x_pos = np.arange(len(bars))

# Create bars and choose color
plt.bar(x_pos, height, color=['darkblue', 'blue', 'cyan'])

# Add title and axis names
plt.title('Accuracy per batchsize')
plt.xlabel('Batchsize')
plt.ylabel('Accuracy')

# Create names on the x axis
plt.xticks(x_pos, bars)

# Show graph
plt.show()

### 4.2 Data augmentation

### Adding the data augmentation to the image data generator


from keras_preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255., validation_split=0.1)

train_generator = datagen.flow_from_dataframe(train,
                                              directory='./',
                                              x_col='image_path',
                                              y_col='mask',
                                              subset='training',
                                              class_mode='categorical',
                                              batch_size=16,
                                              shuffle=True,
                                              target_size=(256,256)
                                             )
valid_generator = datagen.flow_from_dataframe(train,
                                              directory='./',
                                              x_col='image_path',
                                              y_col='mask',
                                              subset='validation',
                                              class_mode='categorical',
                                              batch_size=16,
                                              shuffle=True,
                                              target_size=(256,256)
                                             )
test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(test,
                                                  directory='./',
                                                  x_col='image_path',
                                                  y_col='mask',
                                                  class_mode='categorical',
                                                  batch_size=16,
                                                  shuffle=False,
                                                  target_size=(256,256)
                                                 )

# **1. BUILDING A CNN CLASSIFICATION MODEL**"""

from keras.models import Sequential
input_shape = (256,256,3)

cnn_model_withBatch = Sequential()
cnn_model_withBatch.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
cnn_model_withBatch.add(BatchNormalization())

cnn_model_withBatch.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
cnn_model_withBatch.add(BatchNormalization())
cnn_model_withBatch.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model_withBatch.add(Dropout(0.25))

cnn_model_withBatch.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn_model_withBatch.add(BatchNormalization())
cnn_model_withBatch.add(Dropout(0.25))

cnn_model_withBatch.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn_model_withBatch.add(BatchNormalization())
cnn_model_withBatch.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model_withBatch.add(Dropout(0.25))

cnn_model_withBatch.add(Flatten())

cnn_model_withBatch.add(Dense(512, activation='relu'))
cnn_model_withBatch.add(BatchNormalization())
cnn_model_withBatch.add(Dropout(0.5))

cnn_model_withBatch.add(Dense(128, activation='relu'))
cnn_model_withBatch.add(BatchNormalization())
cnn_model_withBatch.add(Dropout(0.5))

cnn_model_withBatch.add(Dense(2, activation='softmax'))

cnn_model_withBatch.compile(loss = 'categorical_crossentropy',
                            optimizer='adam',
                            metrics= ["accuracy"]
                             )
cnn_model_withBatch.summary()

earlystopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=15
                             )
checkpointer = ModelCheckpoint(filepath="cnn-weights.hdf5",
                               verbose=1,
                               save_best_only=True
                              )
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=10,
                              min_delta=0.0001,
                              factor=0.2
                             )
callbacks = [checkpointer, earlystopping, reduce_lr]

h_cnn = cnn_model_withBatch.fit(train_generator,
                            steps_per_epoch= train_generator.n // train_generator.batch_size,
                            epochs = 50,
                            validation_data= valid_generator,
                            validation_steps= valid_generator.n // valid_generator.batch_size,
                            callbacks=[checkpointer, earlystopping])

# saving model achitecture in json file
model_json = cnn_model_withBatch.to_json()
with open("cnn-model.json", "w") as json_file:
    json_file.write(model_json)

h_cnn.history.keys()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(h_cnn.history['loss']);
plt.plot(h_cnn.history['val_loss']);
plt.title("CNN Classification Model LOSS");
plt.ylabel("loss");
plt.xlabel("Epochs");
plt.legend(['train', 'val']);

plt.subplot(1,2,2)
plt.plot(h_cnn.history['accuracy']);
plt.plot(h_cnn.history['val_accuracy']);
plt.title("CNN Classification Model Accuracy");
plt.ylabel("Accuracy");
plt.xlabel("Epochs");
plt.legend(['train', 'val']);

### 4.3 Classification Model CNN  Evaluation

### Test accuracy:


_, acc = cnn_model_withBatch.evaluate(test_generator)
print("Test accuracy of CNN model : {} %".format(acc*100))

prediction = cnn_model_withBatch.predict(test_generator)

pred = np.argmax(prediction, axis=1)
#pred = np.asarray(pred).astype('str')
original = np.asarray(test['mask']).astype('int')

accuracy = accuracy_score(original, pred)
print("Accuracy of Test Data through CNN is: ",accuracy)

cm = confusion_matrix(original, pred)

report = classification_report(original, pred, labels = [0,1])
print(report)
print("Confusion Matrix of CNN model")
plt.figure(figsize = (5,5))
sns.heatmap(cm, annot=True);


def train_generator(data_frame, batch_size, aug_dict,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    save_to_dir=None,
                    target_size=(256, 256),
                    seed=1):
    image_dg = ImageDataGenerator(**aug_dict)
    mask_dg = ImageDataGenerator(**aug_dict)

    image_generator = image_dg.flow_from_dataframe(
        data_frame,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        x_col="image_path",
        class_mode=None,
        seed=seed)

    mask_generator = mask_dg.flow_from_dataframe(
        data_frame,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        x_col="mask_path",
        class_mode=None,
        seed=seed)

    train_gn = zip(image_generator, mask_generator)

    for (pic, mask) in train_gn:
        pic, mask = datachanges(pic, mask)
        yield (pic, mask)


def datachanges(pic, mask):
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    pic = pic / 255
    return (pic, mask)
