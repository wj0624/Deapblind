import numpy as np
import json
import os
from PIL import Image
from utils.ssd_data import BaseGTUtility

class GTUtility(BaseGTUtility):
    def __init__(self, json_path, image_path, json_file_name, data_folder_name, 
                validation=False, only_with_label=True):
        
        self.json_path = os.path.join(json_path, json_file_name)
        self.image_path = os.path.join(image_path, data_folder_name)
        self.classes = ['Background', 'Text']
        
        self.image_names = []
        self.data = []
        self.text = []
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        for image_name, annotations in gt_data.items():
            img = Image.open(os.path.join(self.image_path, image_name))
            img_width, img_height = img.size

            boxes = [self.get_box(a, img_width, img_height) for a in annotations if 'bbox' in a]
            texts = [a['text'] for a in annotations if 'text' in a]

            if boxes:
                self.image_names.append(image_name)
                self.data.append(np.array(boxes))
                self.text.append(texts)

        self.init()

    @staticmethod
    def get_box(annotation, img_width, img_height):
        x, y, w, h = annotation['bbox']
        box = np.array([x, y, x+w, y+h]) / [img_width, img_height, img_width, img_height]
        return np.append(box, 1)