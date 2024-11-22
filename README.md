<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="https://github.com/user-attachments/assets/1e43be8f-f1e8-43ea-8788-925fd9b4afb3" alt="Logo" width="170" height="170">
  <h1>MEDIBOT</h1>
  <h3>인공지능 기반 의약품 정보 제공 챗봇 서비스</h3>
</div>
</br>

<!-- ABOUT THE PROJECT -->
# 💡About The Project
<h3><a href="https://youtu.be/L_aDKJdg4to">View Demo Here</a></h3>

<strong>의약품 정보 제공 챗봇인 `MEDIBOT`은 의약품, 병원, 약국에 대한 정보를 질문과 답변을 통해 제공합니다.</strong> 
<br/><br/>
<img src = "https://github.com/user-attachments/assets/a6825651-d0be-4868-8b44-18aa2a7931c0" width = "393" hight="261"> 
<img src = "https://github.com/user-attachments/assets/d64f670b-1bf7-43b5-9bfb-07c70c7c2eec" width = "432" hight="287"> 
<br/><br/>
#  📝Description
<br/>
* 포장지나 설명서가 분실된 알약을 눈으로 식별하고 구별하는 것은 굉장히 어려운 일입니다. <br/><br/>
* 알약을 식별하기 위해서는 알약의 여러 식별 조건을 직접 대조하고 확인해야 하는 번거로움이 있습니다. <br/><br/>
* 이를 해결하기 위해, 알약의 사진을 촬영해 전송하면 해당 알약을 자동으로 식별하고 관련 정보를 함께 제공하는 기능을 만들어 보았습니다.<br/><br/>
* 또한 사용자가 특정 약에 대한 정보 혹은 병원과 약국에 대한 정보를 물었을 때, 질문에 대한 구체적이고 직접적인 정보를 제공하는 기능을 추가했습니다. 
<br/>
<br/>

# 📄Feature Description
</br>
<img src = "https://github.com/user-attachments/assets/6f164618-a421-4e75-90df-58fee716113a" width = "600" hight="388"> 
<br/><br/>사용자가 알약 사진을 전송하면, 해당 알약을 식별하여 정보를 제공합니다.<br/><br/>
<img src = "https://github.com/user-attachments/assets/1542a6c5-bded-42a7-803e-d7db4f6a4576" width = "600" hight="388"> 
<br/><br/>사용자가 약국과 병원에 대한 정보를 질문하면 지역에 따른 약국과 병원 목록 그리고 위치 정보를 제공합니다.<br/><br/>
<img src = "https://github.com/user-attachments/assets/d64f670b-1bf7-43b5-9bfb-07c70c7c2eec" width = "600" hight="388"> 
<br/><br/>사용자가 의약품에 대한 구체적인 정보를 질문하면 답변을 제공합니다. 
<br/><br/>


# 🔦 Developing Flow

</br>

<h2>Part1. Pill Classification</h2> 

<h3> Step1. Object Detection & Pill Shape Classifiacation  </h3>

* model : <a href = "https://github.com/ultralytics/yolov5"> YOLOv5 </a>
* train : data augmentation - 5000여장 학습
* precision : 97.9% 
* mAP50 : 99.4 %
* mAP50-95 : 63%

<img src = "https://github.com/user-attachments/assets/e65b1748-e2d6-47b0-8948-89a338edc5dd" width = "450" hight="292"> 
<img src = "https://github.com/user-attachments/assets/e5964f85-1649-419e-9453-ad0f5161e575" width = "450" hight="292"> 


<h3>Step2. Text Detection </h3>

<br>

<img src = "https://github.com/user-attachments/assets/74b407d6-a4ba-43eb-b491-cd3138cd9104" width = "390" hight="240"> 
<img src = "https://github.com/user-attachments/assets/8055949e-c076-4107-91f1-d3f1974f8165" width = "350" hight="220"> 

<br>

* model : <a href = "https://github.com/clovaai/CRAFT-pytorch"> Clova AI CRAFT </a>
* train : 800여장
* data preprocessing
  * sharpening
  * equalization(블록 단위 균등화) 
  * gaussian Blur
    
<br>
  
<img src = "https://github.com/user-attachments/assets/6ed47bd8-f789-49d5-80ea-7390af1d597e" width = "650" hight="410"> 

<br>

* CRAFT 논문 중 model output으로 나온 두가지 산출물 가운데 Region score map만 사용.
* Affinity map은 사용하지 않음. 

<br>

<img src = "https://github.com/user-attachments/assets/92872561-d38c-43bf-ba46-0137950b9e5b" width = "650" hight="410"> 

<br>

* Region score map을 통해 선정된 bounding box를 바탕으로 글자 영역 검출

<br>

<img src = "https://github.com/user-attachments/assets/82b5d064-5858-44c2-aad4-ab7b4e372d57" width = "650" hight="410"> 

<br>

* 검출된 글자 영역을 이미지 상 순서에 맞게 정렬

<br>

<h3>Step3. Text Recognition</h3>

* model : <a href = "https://github.com/clovaai/deep-text-recognition-benchmark"> CLOVA AI Deep-Text-Recognition-Benchmark </a>
* train : 2000여 장의 data
* accuracy : 95%

<h3>Step4. Pill Search </h3> 

* 낱알 식별 정보 database 검색 : <a href = "https://nedrug.mfds.go.kr/pbp/CCBGA01/getItem?totalPages=4&limit=10&page=2&&openDataInfoSeq=11"> 의약품 안전나라 공공 data </a> 

<br>

<h2>Part2. Chatbot Conversation : Natural Language Processing </h2>

 <h3> Step 1. Intent classification </h3>
<img src = "https://github.com/user-attachments/assets/7976ee44-8565-4439-b9af-c0f0ff1a98a5" width = "600" hight="388"> 

* data preprocessing :
  * 형태소 분석 (Konlpy - Okt)
  * 어근 추출
  * 불용어 처리

* model : cnn model
* train : 2500개의 문장 학습
* validation : 500개 문장
* accuracy : 98%

 <h3>Step 2. Named Entity Recognization</h3>

* model : <a href = "https://github.com/Beomi/KcBERT"> KcBert Model Fine Tuning </a>
* train : 2500개의 문장 학습
* validation : 500개 문장

</br></br>
  
# 🗂️ Train Data Example

<img src = "https://github.com/user-attachments/assets/9b0a1e16-edfc-492c-aa27-77d4fd0b3db0" width = "300" hight="550"> 
<img src = "https://github.com/user-attachments/assets/e4f40cd3-811c-4df3-9c91-92a5a148f480" width = "300" hight="550"> 

<br/><br/>

# 💻 Built With

* ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
* ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
* <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
* <img src="https://img.shields.io/badge/django-092E20?style=for-the-badge&logo=django&logoColor=white">
* <img src="https://img.shields.io/badge/mysql-4479A1?style=for-the-badge&logo=mysql&logoColor=white">
* ![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)

<br/>

# 📝Project Flow
<img src = "https://github.com/user-attachments/assets/04de7a68-8660-4f63-8dea-212c06c94382" width = "800" hight="1200"> 

</br></br>

<!-- CONTRIBUTING -->
# 👥 Contributors

<div align="left">
  
* 김민영 <a href ="https://github.com/myqewr"> @myqewr</a>
  * 문장 data 수집 & labeling
  * 알약 사진 data 수집 & labeling
  * Intent Classification / Named Enitity Recognization modeling & 학습
  * Pill Object Detection/Pill Shape Classification/ Text Detection/ Text Recognization modeling & 학습
  * 사용자 질문, 전송된 사진 정보 인식 처리 및 응답 전송을 위한 django-python API backend server 개발
  * django-python backend API server 배포
  * ui/ux 디자인

* 이지민 <a href ="https://github.com/dlwlals1289"> @dlwlals1289 </a>
  * 문장 data 수집
  * 알약 사진 data 수집
  * React Frontend web server 개발
  * client 통신 및 챗봇 서버와의 통신을 위한 spring boot backend server 개발
  * spring boot backend server 배포
  * 프로젝트 기록/관리
  
</div>

</br>

## ETC

### <NER\> pretrained model weight
<a href="https://drive.google.com/file/d/1DB6O_-gO3Y9fd_0SlJObwmIFnOh-vNi-/view?usp=drive_link">download here!</a>
#### project structer for `epoch=4-val_loss=0.01.ckpt`
    
    ├── ...
    ├── chatbot_conv
        ├── intent_classification
        ├── ner 
            ├── model
                ├── epoch=4-val_loss=0.01.ckpt              
            ├── data            
        ├── config           
        └── ... 
    ├── pill_classification
    └── ...


<!-- CONTACT -->
# 🪪 Contact

김민영 MINYOUNG KIM : mygongjoo@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>
