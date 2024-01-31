# This Python file uses the following encoding: cp949
import re
from flask import Flask, request, render_template, jsonify
from flask_restful import Resource, Api, reqparse, abort
from prediction import crop_images, predict_images, preprocessing
from werkzeug.utils import secure_filename


app = Flask(__name__)

path = './data/'
weights_path = 'models/weights.004.h5'

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET','POST'])
def uploader_file():
    if request.method=='POST':
        f = request.files.get("file")
        if f:
            f.save(path + secure_filename(f.filename))
            preprocessing(path + secure_filename(f.filename))
            crop_images(path, "morphology.jpg")
            #crop_images(path, secure_filename(f.filename))

            predictions = predict_images(path, weights_path)
            result_text = " ".join(predictions)

            words = result_text.split()
            for word in words:
                filtered_word = re.sub(r'[^��-�RA-Za-z0-9]', '', word)
            print("filtered_word : "+filtered_word+"\n")
            print("result text : "+result_text+"\n")

            if result_text:
                return jsonify({'result': result_text})
            else:
                return jsonify({'result': '��ȿ���� ���� ����Դϴ�. �ٽ� �õ����ּ���.'})
        else :
            return jsonify({'result': '������ ���޵��� �ʾҽ��ϴ�.'})
    else:
        return "���ٿ� �����߽��ϴ�."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug = True)