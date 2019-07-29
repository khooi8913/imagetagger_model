import base64
import time

import cv2
import mmcv
import torch
from werkzeug import secure_filename
import numpy as np
from torch import FloatTensor as FT

alpha = 0.4
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def buildmodel(modelfile):
    model = torch.load(modelfile, map_location='cpu')
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     model = torch.load(modelfile)
    # else:
    #     model = torch.load(modelfile, map_location='cpu')
    return model


def make_prediction(image, target_size, batch_size, model):
    def predict_x(batch_x, model):
        """
        预测一个batch的数据
        """
        batch_x = batch_x.astype(np.float64)
        batch_x = batch_x / 255.0
        batch_x -= np.ones(batch_x.shape) * mean
        batch_x /= np.ones(batch_x.shape) * std
        # print(batch_x.shape)
        # print(batch_x.dtype)
        batch_x = batch_x.transpose(0, 3, 1, 2)
        # batch_x=batch_x.astype(np.float64)
        batch_x = FT(batch_x)
        with torch.no_grad():
            batch_y = model(batch_x)
        # if torch.cuda.is_available():
        #     with torch.cuda.device(0):
        #         batch_x = batch_x.cuda()
        #         with torch.no_grad():
        #             batch_y = model(batch_x)
        #             # batch_y=Sig(batch_y)
        # else:
        #     with torch.no_grad():
        #         batch_y = model(batch_x)
        batch_y = batch_y.argmax(dim=1)
        batch_y = batch_y.cpu().numpy()

        #     batch_y=batch_y.transpose(0,2,3,1)
        #
        return batch_y

    def update_prediction_center(one_batch):
        wins = []
        for row_begin, row_end, col_begin, col_end in one_batch:
            win = image[row_begin:row_end, col_begin:col_end, :]
            win = np.expand_dims(win, 0)
            wins.append(win)
        x_window = np.concatenate(wins, 0)
        y_window = predict_x(x_window, model)  # 预测一个窗格
        #     print(len(wins))
        for k in range(len(wins)):
            row_begin, row_end, col_begin, col_end = one_batch[k]
            result[row_begin:row_end, col_begin:col_end] = y_window[k]

    batchs = []
    batch = []
    result = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    for row_begin in range(0, image.shape[0], target_size[0]):
        for col_begin in range(0, image.shape[1], target_size[1]):
            row_end = row_begin + target_size[0]
            col_end = col_begin + target_size[1]
            if row_end <= image.shape[0] and col_end <= image.shape[1]:
                batch.append((row_begin, row_end, col_begin, col_end))
                if len(batch) == batch_size:
                    batchs.append(batch)
                    batch = []
    if len(batch) > 0:
        batchs.append(batch)
        batch = []
    for bat in batchs:
        update_prediction_center(bat)
    return result


def createimgs(img, batch_y):
    percentage = batch_y.sum() * 100 / (batch_y.shape[0] * batch_y.shape[1])
    batch_y[batch_y == 1] = 255
    color_mask = np.zeros((batch_y.shape[0], batch_y.shape[1], 3), dtype='uint8')
    color_mask[:, :, 0] = batch_y
    #     color_mask=cv2.resize(color_mask,resolution)
    out = cv2.addWeighted(color_mask, alpha, img, 1 - alpha, 0, img)
    out = cv2.putText(out, "Disease percentage={0:.2f}%".format(percentage), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3,
                      (255, 0, 0), 2)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


def process_prediction(image, modelfile):
    # load the model
    print("DEBUG", "loading the model")
    model = buildmodel(modelfile)
    model.eval()

    # save the image into a temp dir
    filename = '/tmp/' + secure_filename(str(time.time()) + '.jpg')
    with open(filename, 'wb') as f:
        for chunk in image.chunks():
            f.write(chunk)

    # read the image and setup colors
    img = mmcv.imread(filename)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # run prediction
    print("DEBUG", "starting prediction")
    start = time.time()
    batch_y = make_prediction(img, (1080, 1920), 1, model)
    end = time.time()
    time_taken = end - start
    print("time_taken", time_taken)

    outimg = createimgs(img, batch_y)
    cv2.imwrite(filename, outimg)

    return filename


def makeb64image(imagefile):
    with open(imagefile, 'rb') as f:
        encodedimg = base64.b64encode(f.read())

    return "data:image/jpg;base64, " + encodedimg.decode()