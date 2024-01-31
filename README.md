# 💡 한양대 융합전자공학부 졸업작품 2023 💡

👨🏻‍🤝‍👨참여자 : 임원*, 홍준*

## 주제 : 딥러닝을 이용한 텍스트 인식 및 시각장애인용 음성안내 시스템 어플리케이션 개발
Text Recognition Using a Deep Learning Model and Development of an Application for Voice Guidance System for the Visually Impaired

### 1. __개요__
해당 시스템은 안드로이드 기반의 어플리케이션으로 스마트폰을 통해 촬영된 사진으로부터 글자를 추출하고 추출된 글자를 음성으로 안내한다. 딥러닝 모델인 CRNN 모델을 학습시켜 텍스트를 추출 및 인식하고 어플리케이션에서는 예측된 텍스트를 Text-to-Speech 라이브러리를 통해 추출된 글자를 음성으로 출력한다.

### 2. __학습 데이터__
📚AI 허브의 ‘한국어 글자체 이미지' 중 'Text in the wild' 데이터 40,000개

### 3. __구현__
![image](https://github.com/wj0624/Deapblind/assets/128574107/4d40f9fb-7d22-4e3b-a4d4-9fe71dba74d9)

🤖 __딥러닝 모델__
* Tensorflow, keras
* EasyOCR API를 사용해 텍스트 영역을 추출하여 CRNN모델에 전달
* 학습된 CRNN 모델을 통해 이미지 속 글자를 인식하여 예측 값을 flask 서버를 통해 전달

💾 __서버__
* flask 프레임워크
* 안드로이드 앱을 통해 전송한 이미지 파일을 모델에 전달하고, 예측값은 json 파일으로 안드로이드 앱에 전달
* 전달된 이미지를 Rotate, 노이즈 제거, Canny 변환 등 전처리
  
📱 __어플리케이션__
* 안드로이드 스튜디오 Java
* 사용자가 카메라 혹은 갤러리를 통해 이미지 선택
* TTS 모듈을 이용하여 동작 및 예측 결과 음성 안내

### 4. __결과__
* 최종모델 학습결과 : __Accuracy 0.8877__
* 어플리케이션 구현
  
![image](https://github.com/wj0624/Deapblind/assets/128574107/4eda7489-ff55-435a-ae04-544f6143c5a6)
![image](https://github.com/wj0624/Deapblind/assets/128574107/1ac66d87-be98-4235-8abb-3ef790facac3)


