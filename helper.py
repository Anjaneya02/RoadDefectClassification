import tensorflow as tf
import constants
from keras.layers import Input,Dense,Flatten
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
#from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from datetime import datetime
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import swish



# Path to your valid datasetIMPORT
# Path to your valid