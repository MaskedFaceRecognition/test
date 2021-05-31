# 최근 백업 : 20210512
# 여기서 get point를 안바꾼거 backup
### 은 내가 수정한 것
## 은 원래 있던거
### import detect_mask packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os

# import necessary packages train 
import numpy as np
import tensorflow as tf
import cv2
import tqdm
from network import Network
import load

from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

import copy


IMAGE_SIZE = 128
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 1e-3

BATCH_SIZE = 16
PRETRAIN_EPOCH = 500

tf.compat.v1.disable_eager_execution()

def train():
    
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
        #model.predict(...)
    

    # variable로 epoch 같은 자료 저장
    global_step = tf.Variable(0, name='global_step', trainable=False)

    epoch = tf.Variable(0, name='epoch', trainable=False)
    # 
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    # cost 함수를 minimize한다. 
    g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)

    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    if tf.train.get_checkpoint_state('./backup'):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, './backup/latest')
    ### 여기 load에서 npy 파일을 가져온다.
    ### 주의 깊게 봐주자
    x_train, x_test = load.load()
    # x_train.shape : (200, 128, 128, 3), x_test.shape : (95, 128, 128, 3)


    x_train_get = copy.deepcopy(x_train)
    x_test_get = copy.deepcopy(x_test)
    x_train = np.array([a / 127.5 - 1 for a in x_train])
    x_test = np.array([a / 127.5 - 1 for a in x_test])

    step_num = int(len(x_train) / BATCH_SIZE)
    while True:
        sess.run(tf.compat.v1.assign(epoch, tf.add(epoch, 1)))
        print('epoch: {}'.format(sess.run(epoch)))

        args = {'image': 'images/pic1.jpeg', 'face': 'face_detector', 'model': 'mask_detector.model', 'confidence': 0.5}
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
            #################################################################
            # load the face mask detector model from disk
            #print("[INFO] loading face mask detector model...")
            get_point_model = load_model(args["model"])
            #################################################################
            for i in tqdm.tqdm(range(step_num)):
                x_batch_get = x_train_get[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                x_batch = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                ### get_points에 x_batch를 넣고 하나씩 넣어줘야 할듯
                points_batch, mask_batch = get_points(x_batch_get,net,args,get_point_model)

                _, g_loss = sess.run([g_train_op, model.g_loss], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                g_loss_value += g_loss

            #print('Completion loss: {}'.format(g_loss_value))
            np.random.shuffle(x_test)

            x_batch = x_test[:BATCH_SIZE]
            #x_batch_get = x_test_get[:BATCH_SIZE]
            print(f'yes?')
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            print(f'yes')
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)

            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, './backup/latest', write_meta_graph=False)

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
            x_batch = x_test[:BATCH_SIZE]ㅎ
            completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
            sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
            cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

            tf.initialize_all_variables()
            saver = tf.compat.v1.train.Saver()

            saver.save(sess, './backup/latest', write_meta_graph=False)


### 여기 부분에서 points와 mask 부분이 정확하게 뭔지 알아야 한다.
### 여기서 get_points와 mask_image(x_batch)를 조절해야 한다.
def get_points(x_batch,net,args,model):
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        image=np.array(x_batch[i])
        
        # load the input image from disk, clone it, and grab the image spatial
        # dimensions
        # 아랫부분 필요 없을듯
        #image = cv2.imread(args["image"])
        #print(image.shape)
        #orig = image.copy()
        image = cv2.resize(image, (128, 128))
        
        (h, w) = image.shape[:2]

        # construct a blob from the image
        ### blobFromImage(image, scalefactor, size, mean, swapRB, crop, ddepth)
        ### 1. input image 신경망을 통과 할 이미지
        ### 2. scalefactor : 선택적으로 특정요소만큼 이미지 크기를 늘리거나 줄임 
        ### 3. size : 신경망이 예상하는 이미지의 크기 
        ## size (300,300)-> (224,224) -> (128 128)
        blob = cv2.dnn.blobFromImage(image,1.0,(128, 128),(104.0, 177.0, 123.0))
        #blob = cv2.dnn.blobFromImage(image,1/255, (224, 224), (0, 0, 0))
        # pass the blob through the network and obtain the face detections
        #print("[INFO] computing face detections...")
        net.setInput(blob)
        detections = net.forward()
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            
            #(startX, startY, endX, endY) = box.astype("int")
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # ensure the bounding boxes fall within the dimensions of
                # the frame
                # 
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                mask_point=[startX, (startY+endY)*19//40, (endX+10), endY+10]
                points.append(mask_point)

                p1,q1,p2,q2 = startX, (startY+endY)*19//40, (endX+10), endY+10
                m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
                m[q1:q2 + 1, p1:p2 + 1] = 1
                mask.append(m)
                break
    print(f"np.array(points.shape) :{np.array(points).shape}, np.array(mask).shape :{np.array(mask).shape} ")
    return np.array(points), np.array(mask)


if __name__ == '__main__':
    train()
