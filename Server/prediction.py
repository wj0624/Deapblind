import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from easyocr import Reader
from tensorflow.keras.optimizers import SGD, Adam
from models.crnn_model import CRNN
from models.crnn_utils import decode
from models.hyper_param import (ocr_dict, leaky, fctc, drop, opt, input_shape, 
                                lr, decay, momentum, clipnorm, freeze)

#이미지 전처리
def preprocessing(imgPath):
    src = cv2.imread(imgPath, 1)
    # grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # canny 엣지 검출
    canned = cv2.Canny(gray, 150, 300)

    # 엣지연결
    kernel = np.ones((10,1),np.uint8) # 가로 1 세로 10
    mask = cv2.dilate(canned, kernel, iterations = 20)

    # contours 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 contours 찾기
    biggest_cntr = None
    biggest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > biggest_area:
            biggest_area = area
            biggest_cntr = contour

    # 외곽 box
    rect = cv2.minAreaRect(biggest_cntr)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 외곽 box 그리기
    src_box = src.copy()
    cv2.drawContours(src_box, [box], 0, (0, 255, 0), 3)

    # angle 계산
    angle = rect[-1]
    if angle > 45:
        angle = -(90 - angle)

    # 기울기 조정
    rotated = src.copy()
    (h, w) = rotated.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(rotated, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 회전된 박스 좌표 찾기
    ones = np.ones(shape=(len(box), 1))
    points_ones = np.hstack([box, ones])
    transformed_box = M.dot(points_ones.T).T

    y = [transformed_box[0][1], transformed_box[1][1], transformed_box[2][1], transformed_box[3][1]]
    x = [transformed_box[0][0], transformed_box[1][0], transformed_box[2][0], transformed_box[3][0]]

    y1, y2 = int(min(y)), int(max(y))
    x1, x2 = int(min(x)), int(max(x))

    # crop
    crop = rotated[y1:y2, x1:x2]

    #흑백처리
    gray2 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)
    filtering = cv2.bilateralFilter(binary, -1, 10, 5)
    kernel = np.ones((3, 1), np.uint8)
    morphology = cv2.morphologyEx(filtering, cv2.MORPH_OPEN, kernel)

    # #mser
    # mser = cv2.MSER_create()
    # regions, _ = mser.detectRegions(gray2)

    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    # for j, cnt in enumerate(hulls):
    #     x, y, w, h = cv2.boundingRect(cnt)

    #     if w > 160:
    #         continue

    #     cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # cv2.imwrite('./data/mser.jpg', dst)

    cv2.imwrite("./data/mask.jpg", mask)
    cv2.imwrite("./data/box.jpg", src_box)
    cv2.imwrite("./data/canny.jpg", canned)
    cv2.imwrite("./data/rotated.jpg", rotated)
    cv2.imwrite("./data/cropped.jpg", crop)
    cv2.imwrite("./data/gray.jpg", gray2)
    cv2.imwrite("./data/binary.jpg", binary)
    cv2.imwrite("./data/filtering.jpg", filtering)
    cv2.imwrite("./data/morphology.jpg", morphology)

def resize_image(img_pil):
    """이미지를 리사이즈하고 패딩을 추가합니다."""
    original_width, original_height = img_pil.size
    new_width = int((original_width / original_height) * 32)
    img_pil = img_pil.resize((new_width, 32))
    
    delta_w = 256 - new_width
    padding = (0, 0, delta_w, 0)
    img_pil = ImageOps.expand(img_pil, padding, fill='black')
    
    return img_pil

def extract_bbox_coordinates(bbox):
    """Bounding Box 좌표를 추출합니다."""
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    
    return tl, tr, br, bl


def crop_images(base_path, image_name):
    """OCR을 사용해 이미지에서 텍스트를 검출하고 해당 부분을 자릅니다."""
    image_path = os.path.join(base_path, image_name)
    image = cv2.imread(image_path)

    langs = ['ko', 'en']
    reader = Reader(lang_list=langs, gpu=True)
    results = reader.readtext(image_path)

    for idx, (bbox, _, _) in enumerate(results):
        tl, _, br, _ = extract_bbox_coordinates(bbox)
        
        cropped = image[tl[1]:br[1], tl[0]:br[0]]
        if (cropped is not None):
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            resized_cropped = resize_image(cropped_pil)
            cropped_filename = f"cropped_img_{idx+1}.jpg"
            cropped_file_path = os.path.join(base_path, cropped_filename)
            resized_cropped.save(cropped_file_path)


def process_image(img_path, model_pred, ocr_dict):
    """CRNN 모델을 사용해 이미지에서 텍스트를 예측합니다."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.normalize(img, img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img = img[:, :, None]

    img_array = img.astype(np.float32)
    img_array = np.transpose(img_array, (1, 0, 2))
    images = np.expand_dims(img_array, axis=0)
    
    pred_data = model_pred.predict(images)
    chars = [ocr_dict[c] for c in np.argmax(pred_data[0], axis=1)]
    pred_str = decode(chars)

    return pred_str


def predict_images(base_path, weights_path):
    """주어진 경로에 있는 모든 이미지에 대해 예측을 수행합니다."""
    model, model_pred = CRNN(input_shape, len(ocr_dict), leaky=leaky, fctc=fctc, drop=drop, gru=False)
    model.load_weights(weights_path)

    if opt == Adam:
        optimizer = Adam(learning_rate=lr, clipnorm=clipnorm)
    else:
        optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=True, clipnorm=clipnorm)

    for layer in model.layers:
            layer.trainable = not layer.name in freeze

    key = 'focal_ctc' if fctc else 'ctc'
    model.compile(loss={key: lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    cropped_files = [f for f in os.listdir(base_path) if f.startswith("cropped_img")]

    predictions = []
    for cropped_file in cropped_files:
        cropped_path = os.path.join(base_path, cropped_file)
        predicted_text = process_image(cropped_path, model_pred, ocr_dict)
        predictions.append(predicted_text)
        os.remove(cropped_path)
    
    return predictions