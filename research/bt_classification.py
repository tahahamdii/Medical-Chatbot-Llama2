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
