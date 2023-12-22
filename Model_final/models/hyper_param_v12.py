from crnn_utils_v1 import alphabet87, korean2350

ocr_dict = korean2350 + alphabet87
input_width, input_height = 256, 32
input_shape = (input_width, input_height, 1)
max_string_len = 62

epoch = 300
batch_size = 128 # 256, 128, 64
leaky = 0.1 # 0, 0.05, 0.1
fctc = 0.25, 0.75 # 0, 0.25, 0.5, 0.75, 1
drop = None # 0, 0.1 
freeze = []# 'conv1_1', 'conv2_1', 'conv3_1', 'conv3_2', 'conv4_1' , 'conv5_1'

opt = 'Adam' # SGD, Adam
lr = 1e-3 # 1e-1, 1e-2, 1e-3, 1e-4, 1e-5
decay = 1e-3 # 1e-2, 1e-3, 1e-4, 1e-5, 1e-6
momentum = 0.9
clipnorm = 1. # 1, 2, 3, 4, 5