# -*- coding: utf-8 -*
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import tensorflow as tf
import numpy as np

import glob
import os

### train parameters
IMAGE_SIZE = 128
LOCAL_SIZE = 64
LEARNING_RATE = 1e-3

BATCH_SIZE = 16
PRETRAIN_EPOCH = 100
tf.compat.v1.disable_eager_execution()

from flask import Flask, url_for, redirect, render_template, request
#from flask_ngrok import run_with_ngrok

app = Flask(__name__)



@app.route("/")
def hello2():
    angry_path = './static/Emotion/angry'
    happy_path = './static/Emotion/happy'
    neutral_path = './static/Emotion/neutral'
    angry_file = os.listdir(angry_path)
    happy_file = os.listdir(happy_path)
    neutral_file = os.listdir(neutral_path)
    angry_num = str(len(angry_file))
    happy_num = str(len(happy_file))
    neutral_num = str(len(neutral_file))
    return render_template("hello2.html",angry_num = angry_num, happy_num = happy_num, neutral_num = neutral_num)

# Emotion - 예측해서 GAN에 활용할 사진(#.jpg) 업로드 or 캡처 
@app.route("/multi_upload_emotion", methods = ['POST'])
def multi_upload_emotion():
    uploaded_files = request.files.getlist("file[]")
    IDX = 0
    for file in uploaded_files:
        file.save("./Web/static/Emotion/{}.jpg".format(IDX))
        IDX += 1
    return redirect(url_for("hello2"))

# GAN - 복원시킬 사진(#.jpg) 업로드
@app.route("/multi_upload_gan", methods = ["POST"])
def multi_upload_gan():
    uploaded_files = request.files.getlist("file[]")
    IDX = 0
    for file in uploaded_files:
        file.save("./static/GAN/{}.jpg".format(IDX))
        IDX += 1
    return redirect(url_for("hello2"))

@app.route("/emotion_prediction", methods = ['POST'])
def emotion_prediction():
    # 1. 이미지 업로드
    # 2. resize
    # 3. 모델 불러와서 업로드한 이미지 학습
    # 4. 예측된 표정에 맞는 곳에 저장(ex)"static/Emotion/angry/{}.jpg")
    uploaded_files = request.files.getlist("file[]")
    index_happy = 0
    index_angry = 0
    index_neutral = 0
    model = load_model('./models/model_best_0_2.h5')
    for file in uploaded_files:
        filestr = file.read() # byte 단위이기 때문에 바로 file.save로 저장해서 .jpg로 보이지 않는다.
        
        detection_model_path = './models/haarcascade_frontalface_default.xml'
        face_detection = cv2.CascadeClassifier(detection_model_path)

        #convert string data to numpy array
        npimg = np.fromstring(filestr, np.uint8)
        # convert numpy array to image
        frame = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE) # imread와 달리 byte 읽기
        saveFile = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # print(type(frame)) # numpy.ndarray
        face = face_detection.detectMultiScale(frame, scaleFactor = 1.1, minNeighbors = 5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
        
        # print(face)
        if len(face) == 0: # 얼굴 인식 안된 사진은 Train Set에 저장되지 않도록
            continue
        face = sorted(face, reverse = True, key = lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face 
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0)
        # print(roi)

        prediction = model.predict(roi) # ex) [happy, angry, neutral] -> [0.33, 0.1, 0.57]
        print(prediction)

        # {'angry': 0, 'happy': 1, 'neutral': 2}
        # Threshold를 0.5로 설정. 확률 0.5가 넘는 표정이 있으면 해당 폴더에 저장.
        if prediction[0][0] >= 0.5:
            cv2.imwrite("./static/Emotion/angry/{}.jpg".format(index_angry), saveFile) # numpy.ndarray
            index_angry += 1
        elif prediction[0][1] >= 0.5:
            cv2.imwrite("./static/Emotion/happy/{}.jpg".format(index_happy), saveFile)
            index_happy += 1
        elif prediction[0][2] >= 0.5:
            cv2.imwrite("./static/Emotion/neutral/{}.jpg".format(index_neutral), saveFile)
            index_neutral += 1
    return redirect(url_for("hello2"))

@app.route('/upload_nonmasked')
def emotion():
    return render_template('upload_nonmasked.html')

@app.route("/upload_masked")
def restore():
    return render_template("upload_masked.html")


################################ 작성 일자 : 20210528
################################ 작성자 : 김동우 수정
################################ 수정 내용 : web과 gan 합치기
################################
################################ 

from src_upgraded.network import Network
from src_upgraded.load import *
import copy
import tqdm
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.backend import set_session



### test 데이터 업로드 ( masked image를 업로드한다. )
@app.route("/upload_test", methods = ["POST"])
def upload_test():
    uploaded_files = request.files.getlist("file[]")
    IDX = 0
    for file in uploaded_files:
        file.save("./data/images/test/{}.jpg".format(IDX))
        IDX += 1
    #return render_template('gan_mask_detect_modeling.html')
    return redirect(url_for("hello2"))

### train 데이터 업로드 ( unmasked image를 업로드한다.)
@app.route("/upload_train", methods = ["POST"])
def upload_train():
    uploaded_files = request.files.getlist("file[]")
    IDX = 0
    for file in uploaded_files:
        file.save("./static/test/{}.jpg".format(IDX))
        IDX += 1
    #return render_template('gan_mask_detect_modeling.html')
    return redirect(url_for("hello2"))

@app.route("/to_npy", methods = ["POST"])
def to_npy(train_photo_path,test_photo_path,save_path):
    image_size=128
    train = []
    test = []
    train_paths = glob.glob(train_photo_path)
    for path in train_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train.append(img)

    test_paths = glob.glob(test_photo_path)
    for path in test_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test.append(img)

    train = np.array(train, dtype=np.uint8)
    test = np.array(test, dtype=np.uint8)

    x_train = train
    x_test = test

    if not os.path.exists('./npy'):
        os.mkdir('./npy')
    np.save(save_path+ '/x_train.npy', x_train)
    np.save(save_path+ '/x_test.npy', x_test)
    return redirect(url_for("hello2"))


@app.route('/train_angry',methods = ["POST"])
def train_angry():
    train_photo_path = './static/Emotion/angry/*'
    save_path = './static/npy/npy_angry'
    test_photo_path = './static/test/*'
    to_npy(train_photo_path,test_photo_path,save_path)
    train(save_path)

@app.route('/train_happy',methods = ["POST"])
def train_happy():
    train_photo_path = './static/Emotion/happy/*'
    save_path = './static/npy/npy_happy/'
    test_photo_path = './static/test/*'
    to_npy(train_photo_path,test_photo_path,save_path)
    train(save_path)

@app.route('/train_neutral',methods = ["POST"])
def train_neutral():
    train_photo_path = './static/Emotion/neutral/*'
    save_path = './static/npy/npy_neutral/'
    test_photo_path = './static/test/*'
    to_npy(train_photo_path,test_photo_path,save_path)
    train(save_path)

def train(npy_path):
    # placeholder로 변수 저장
    x = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])

    local_completion = tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.compat.v1.placeholder(tf.bool, [])

    model = Network(x, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)
    sess = tf.compat.v1.Session()

    
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        set_session(sess)
    
    # variable로 epoch 같은 자료 저장
    global_step = tf.Variable(0, name='global_step', trainable=False)
    epoch = tf.Variable(0, name='epoch', trainable=False)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    # cost 함수를 minimize한다. 
    g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)
    #init_op=tf.compat.v1.initialize_all_variables()
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    saver = tf.compat.v1.train.Saver()
    if tf.train.get_checkpoint_state('./src_upgraded/backup'):
        #saver = tf.compat.v1.train.Saver()
        saver.restore(sess, './src_upgraded/backup/latest')
    ### 여기 load에서 npy 파일을 가져온다.
    x_train, x_test = load_data(npy_path)
    # x_train.shape : (200, 128, 128, 3), x_test.shape : (95, 128, 128, 3)


    x_train_get = copy.deepcopy(x_train)
    x_test_get = copy.deepcopy(x_test)
    x_train = np.array([a / 127.5 - 1 for a in x_train])
    x_test = np.array([a / 127.5 - 1 for a in x_test])

    step_num = int(len(x_train) / BATCH_SIZE)
    while True:
        sess.run(tf.compat.v1.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))

        args = {'face': 'src_upgraded/face_detector', 'model': 'src_upgraded/mask_detector.model', 'confidence': 0.5}
        #print(f'args : {args}') ###
        # load our serialized face detector model from disk
        #print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
        weightsPath = os.path.sep.join([args["face"],
            "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNet(prototxtPath, weightsPath)

        np.random.shuffle(x_train)
        if sess.run(epoch) <= PRETRAIN_EPOCH:
            g_loss_value = 0
            get_point_model = load_model(args["model"])
            for i in tqdm.tqdm(range(step_num)):
                x_batch_get = x_train_get[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                ### get_points에 x_batch를 넣고 하나씩 넣어줘야 할듯
                points_batch, mask_batch = get_points(x_batch_get,net,args,get_point_model)
                _, g_loss = sess.run([g_train_op, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                g_loss_value += g_loss

            np.random.shuffle(x_test)

            x_batch = x_test[:BATCH_SIZE]
            #x_batch_get = x_test_get[:BATCH_SIZE]
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)

            cv2.imwrite('./src_upgraded/output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
            saver.save(sess, './src_upgraded/backup/latest', write_meta_graph=False)
            if sess.run(epoch) == PRETRAIN_EPOCH:
                saver.save(sess, './backup/pretrained', write_meta_graph=False)
        # Discrimitation
        else:
            g_loss_value = 0
            d_loss_value = 0


            # 퍼센트바 만들어짐
            for i in tqdm.tqdm(range(step_num)):
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                x_batch_get = x_train_get[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                ### 이 자리에 get_points에서 무엇을 가져오나
                points_batch, mask_batch = get_points(x_batch_get,net,args,model)
                _, g_loss, completion = sess.run([g_train_op, model.g_loss, model.completion], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                g_loss_value += g_loss
                local_x_batch = []
                local_completion_batch = []

                ### 아래 for문에서 조리해야함
                for i in range(BATCH_SIZE):
                    ### 포인트 배치에서 받은 값을 여기에 저장한다.
                    x1, y1, x2, y2 = points_batch[i]
                    local_x_batch.append(x_batch[i][y1:y2, x1:x2, :])
                    local_completion_batch.append(completion[i][y1:y2, x1:x2, :])

                local_x_batch = np.array(local_x_batch)
                local_completion_batch = np.array(local_completion_batch)

                _, d_loss = sess.run(
                    [d_train_op, model.d_loss], 
                    feed_dict={x: x_batch, mask: mask_batch, local_x: local_x_batch, global_completion: completion, local_completion: local_completion_batch, is_training: True})
                d_loss_value += d_loss

            print('Completion loss: {}'.format(g_loss_value))
            print('Discriminator loss: {}'.format(d_loss_value))

            np.random.shuffle(x_test)
            x_batch = x_test[:BATCH_SIZE]
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./src_upgraded/output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

            saver.save(sess, './src_upgraded/backup/latest', write_meta_graph=False)


def get_points(x_batch,net,args,model):
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        image=np.array(x_batch[i])
        image = cv2.resize(image, (128, 128))
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        points.append([x1, y1, x2, y2])
        (h, w) = image.shape[:2]

        blob = cv2.dnn.blobFromImage(image,1.0,(128, 128),(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args["confidence"]:
                p1,q1,p2,q2 = max(0,startX-20), (startY+endY)*15//40, (endX+10), endY+10
                m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
                m[q1:q2 + 1, p1:p2 + 1] = 1
                mask.append(m)
                break

    return np.array(points), np.array(mask)


if __name__ == "__main__":
    face_detection = load_detection_model('models/haarcascade_frontalface_default.xml')
    model = load_model('models/model_best_0_2.h5') # model load
    #app.run(host='0.0.0.0') # 외부에서 접근가능한 서버로 만들어준다, 외부에서 접근가능하도록 하는 U
    app.run(debug = True)
    app.run() # defalut port = 5000
