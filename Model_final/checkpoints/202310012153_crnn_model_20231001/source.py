# In[0]


# In[1]
import os
import time
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import font_manager
from livelossplot.keras import PlotLossesCallback
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from crnn_model_v1 import CRNN
from crnn_data_v1 import InputGenerator, modified_gen
from utils.training import Logger
from utils.gt_util import GTUtility

# In[2]
devices = device_lib.list_local_devices()
device_types = [device.device_type for device in devices]
if 'GPU' in device_types:
    print("Current device type:           GPU")
else:
    print("Current device type:           CPU")

current_directory = os.getcwd()
os.chdir(current_directory)
print(f"Current working directory:    ", os.getcwd())

font_path = 'fonts/NanumBarunGothic.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['font.size'] = 12
plt.rcParams['font.sans-serif'] = ['NanumBarunGothic']
print("Current font name:            ", plt.rcParams['font.family'])

# In[3]
base_path = 'data/text_in_the_wild_data/'
model_name = 'crnn_model_20231001'
from models.hyper_param_v12 import (ocr_dict, leaky, fctc, drop, batch_size, input_shape, max_string_len, 
                        lr, decay, momentum, clipnorm, epoch, freeze, opt)

print(f"Model Name: {model_name.upper()}")
print("---------------------------------")
print("Model Type: CRNN")
print("Model Param:")
print(f"    epoch:         {epoch}")
print(f"    batch_size:    {batch_size}")
print(f"    leaky ReLU:    {leaky}")
print(f"    focal CTC:     {fctc}")
print(f"    dropout:       {drop}")
print(f"    freeze:        {freeze if freeze else 'None'}")
print("---------------------------------")
print(f"Optimizer: {opt}")
print("Optimizer Param:")
print(f"    learning rate: {lr}")
print(f"    decay:         {decay}")
print(f"    momentum(SGD): {momentum}")
print(f"    clipnorm:      {clipnorm}")

# In[4]
with open(base_path + 'gt_train_util_text.pkl', 'rb') as f:
    gt_train_util = pickle.load(f)
gen_train = InputGenerator(gt_train_util, batch_size, ocr_dict, input_shape[:2], 
                        grayscale=True, max_string_len=max_string_len, concatenate=False, fctc=fctc)
print("Total number of train data samples:        ", gen_train.num_samples)

with open(base_path + 'gt_val_util_text.pkl', 'rb') as f:
    gt_val_util = pickle.load(f)
gen_val = InputGenerator(gt_val_util, batch_size, ocr_dict, input_shape[:2], 
                        grayscale=True, max_string_len=max_string_len, concatenate=False, fctc=fctc)
print("Total number of validation data samples:   ", gen_val.num_samples)

# In[5]
model, model_pred = CRNN(input_shape, len(ocr_dict), leaky=leaky, fctc=fctc, drop=drop, gru=False)
#model_path = 'models/'
#model_name = 'leaky_fctc_weights.044.h5'
#model.load_weights(model_path + model_name)

if opt == 'Adam': 
        optimizer = Adam(learning_rate=lr, decay=decay, clipnorm=clipnorm)
else: optimizer = SGD(learning_rate=lr, decay=decay, clipnorm=clipnorm, momentum=momentum, nesterov=True)

for layer in model.layers:
        layer.trainable = not layer.name in freeze

key = 'focal_ctc' if fctc else 'ctc'
model.compile(loss={key: lambda y_true, y_pred: y_pred}, optimizer=optimizer, metrics=['accuracy'])
model.summary()

# In[6]
experiment = model_name
checkdir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + experiment
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())

