import argparse
from pathlib import Path

import sys
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import datetime
# from .event import *
from PV_sent_oss import MyThread
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# import websockets
# import asyncio
# from table import save_object
# from models import save_object
# from concurrent.futures import ThreadPoolExecutor,as_completed,wait
from settings import *

from PV_get_rtmp import RTMPLiveStream

import threading
import multiprocessing
from mask_fix import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

actions_name_dict = {
    "action1": 'helmet',
    "action2": 'no_helmet',
    "action3": 'reflective_vest',
    "action4": 'no_reflective_vest',
    "action5": "fgbx_normal",
    "action6": 'dangerous_invasion',
    "action7": 'dangerous_invasion_normal',
    "action8": "smoke",
    "action81": "normal_smoke",
    "action9": "person",
    "action10": "hand",
    "equipment": 'AI-RebarFactory1'
}

algor_dict = {
    '01': {
        'description': '钢筋工厂吊装作业检测',
        '011': '钢筋吊篓数量超过2个',
        '012': '作业陪同人员不足2人'
    },
    '02': {
        'description': '安全帽、反光背心佩戴检测',
        '021': '未佩戴安全帽',
        '022': '未佩戴反光背心'
    },
    '03': {
        'description': '危险区域入侵检测',
        '031': '存在人员进入危险区域',
    },
    '04': {
        'description': '吸烟检测',
        '041': '存在人员吸烟'
    },

}

# 定义新的需求，在如果没有人的情况下，每隔多少时间给他们发送一张正常的检测图片
save_count_capture = 250
output_path = get_output_path()
logger_path = get_logger_path()
interval = 3
global capture
count_frame = 0
# 每隔多少时间发送一张正常图片
normal_sent_time = 1800

daytime = is_daytime()


def detect(save_img=False):
    global opt
    global count_frame, frames_action
    global lable_action, video_save_path, out_writer
    global fourcc, video_fps, size, deque_action, action2_w_count, action2_w1_count, img_list_b,  actions_name_dict, save_count_capture
    global t_now, t_now1
    global device_id, son_device_id
    global action_smoke_save_key

    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # 模型中的参数名称： names== == ['worker', 'hook', 'steel', 'two_steel']
    # 消息参数添加===========
    if actions_name_dict["action1"] in names:
        MessageWarn_dict["algorithmId"] = int("02")
        MessageWarn_dict["algorithmName"] = algor_dict["02"]["description"]

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    # 变量定义
    sum_count = 0
    # 获取当天白天的时间：================
    now1 = datetime.datetime.now()
    morning_start = datetime.datetime(now1.year, now1.month, now1.day, 6, 0, 0)
    morning_end = datetime.datetime(now1.year, now1.month, now1.day, 18, 0, 0)
    # # =================================================================
    #吸烟参数逻辑区定义
    # person_head_info = []
    # hand_info = []

    #开关打开条件
    action_interval_count = 0
    count_action_list = []


    #保存条件开关
    action_smoke_save_key = False
    action_smoke_save_key_count = 0


    #非手和头的动作计数器--------------
    no_action_count = 0


    #im0暂存列表
    img_list = []
    img_list_b = []



    # ========================================

    #=====================画图参数区域设定=========================

    hl1 = 1080 / 1080  # 监测区域高度距离图片顶部比例
    wl1 = 1 / 1920  # 监测区域高度距离图片左部比例

    hl2 = 437 / 1080  # 监测区域高度距离图片顶部比例
    wl2 = 225 / 1920  # 监测区域高度距离图片左部比例

    hl3 = 389 / 1080  # 监测区域高度距离图片顶部比例
    wl3 = 1763 / 1920  # 监测区域高度距离图片左部比例

    hl4 = 1080 / 1080  # 监测区域高度距离图片顶部比例
    wl4 = 1918 / 1920  # 监测区域高度距离图片左部比例

    #===========================================================
    t_now = time.time()
    print("时间0----", t_now)
    # 设置跳帧参数
    skip_frames = 1  # 跳过的帧数，例如 2 表示每处理 2 帧后跳过 1 帧
    for path, img, im0s, vid_cap in dataset:
        # 定义时间检测参数：t_now,t_now1
        sum_count += 1
        print("sum_count:===", sum_count)
        if sum_count % skip_frames == 0:
            # ========================设置检测区域================================================================================
            if webcam:
                for b in range(0, img.shape[0]):
                    mask = np.zeros([img[b].shape[1], img[b].shape[2]], dtype=np.uint8)
                    # mask[round(img[b].shape[1] * hl1):img[b].shape[1], round(img[b].shape[2] * wl1):img[b].shape[2]] = 255
                    pts = np.array([[int(img[b].shape[2] * wl1), int(img[b].shape[1] * hl1)],  # pts1
                                    [int(img[b].shape[2] * wl2), int(img[b].shape[1] * hl2)],  # pts2
                                    [int(img[b].shape[2] * wl3), int(img[b].shape[1] * hl3)],  # pts3
                                    [int(img[b].shape[2] * wl4), int(img[b].shape[1] * hl4)],
                                    ], np.int32)
                    mask = cv2.fillPoly(mask, [pts], (255, 255, 255))
                    imgc = img[b].transpose((1, 2, 0))
                    imgc = cv2.add(imgc, np.zeros(np.shape(imgc), dtype=np.uint8), mask=mask)
                    # cv2.imshow('1',imgc)
                    img[b] = imgc.transpose((2, 0, 1))
            else:
                mask = np.zeros([img.shape[1], img.shape[2]], dtype=np.uint8)
                # mask[round(img.shape[1] * hl1):img.shape[1], round(img.shape[2] * wl1):img.shape[2]] = 255
                pts = np.array([[int(img.shape[2] * wl1), int(img.shape[1] * hl1)],  # pts1
                                [int(img.shape[2] * wl2), int(img.shape[1] * hl2)],  # pts2
                                [int(img.shape[2] * wl3), int(img.shape[1] * hl3)],  # pts3
                                [int(img.shape[2] * wl4), int(img.shape[1] * hl4)],

                                ], np.int32)
                mask = cv2.fillPoly(mask, [pts], (255, 255, 255))
                img = img.transpose((1, 2, 0))
                img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
                img = img.transpose((2, 0, 1))
            # =================================================================================

            # =========================上面注释掉就是不设置检测区域框============================================
            # 这里是将读出来的图片数据进行通道数修改====
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            # 这里将图片数据放入模型进行预测 这里的预测结果为六个值得
            pred = model(img, augment=opt.augment)[0]
            # print("这里是刚将图片数据放入pred进行预测的第一次结果：", pred.shape)

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@",t2-t1)
            # print("pred:",pred)
            # Apply Classifier
            t11 = time.time()
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            # Process detections
            w, h = 0, 0

            hand_info = []
            person_head_info =[]
            for i, det in enumerate(pred):  # detections per image
                res = []
                # # print("i:det",det)
                # if webcam:  # batch_size >= 1
                #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                # else:
                #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                # ================将检测区域的范围用线条描绘出来===============================================

                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                    is_black = is_image_fully_black(im0)
                    if is_black == True:
                        sys.exit()

                    # cv2.putText(im0, (int(im0.shape[1] * wl1 - 5), int(im0.shape[0] * hl1 - 5)),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1.0, (255, 255, 0), 2, cv2.LINE_AA)
                    #
                    # pts = np.array([[int(im0.shape[1] * wl1), int(im0.shape[0] * hl1)],  # pts1
                    #                 [int(im0.shape[1] * wl2), int(im0.shape[0] * hl2)],  # pts2
                    #                 [int(im0.shape[1] * wl3), int(im0.shape[0] * hl3)],  # pts3
                    #                 [int(im0.shape[1] * wl4), int(im0.shape[0] * hl4)],
                    #                 ], np.int32)  # pts4
                    # # [int(im0.shape[1] * wl5), int(im0.shape[0] * hl5)]
                    # # pts = pts.reshape((-1, 1, 2))
                    # zeros = np.zeros((im0.shape), dtype=np.uint8)
                    # mask = cv2.fillPoly(zeros, [pts], color=(0, 165, 255))
                    # im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)
                    # cv2.polylines(im0, [pts], True, (255, 255, 0), 3)
                    # # plot_one_box(dr, im0, label='Detection_Region', color=(0, 255, 0), line_thickness=2)

                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                    # # print("这时未遮蔽时的图片矩阵", im0)
                    # cv2.putText(im0, "Detection_Region", (int(im0.shape[1] * wl1 - 5), int(im0.shape[0] * hl1 - 5)),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1.0, (255, 255, 0), 2, cv2.LINE_AA)
                    # pts = np.array([[int(im0.shape[1] * wl1), int(im0.shape[0] * hl1)],  # pts1
                    #                 [int(im0.shape[1] * wl2), int(im0.shape[0] * hl2)],  # pts2
                    #                 [int(im0.shape[1] * wl3), int(im0.shape[0] * hl3)],  # pts3
                    #                 [int(im0.shape[1] * wl4), int(im0.shape[0] * hl4)],
                    #                 ], np.int32)  # pts4
                    # # [int(im0.shape[1] * wl5), int(im0.shape[0] * hl5)]
                    # # pts = pts.reshape((-1, 1, 2))
                    # zeros = np.zeros((im0.shape), dtype=np.uint8)
                    # mask = cv2.fillPoly(zeros, [pts], color=(0, 165, 255))
                    # im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)
                    #
                    # cv2.polylines(im0, [pts], True, (255, 255, 0), 3)

                # ============================检测区域范围描框=================================================
                count_frame = frame
                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else '_{}'.format(frame))  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += "{} {}{}, ".format(n, names[int(c)], 's' * (n > 1))  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')


                        if save_img or view_img:  # Add bbox to image
                            if cls == 0:
                                person_head_info.append(xyxy)
                            elif cls == 1:
                                hand_info.append(xyxy)
                            # print("cls====",cls)
                            if cls == 0 or cls == 1:
                                label = names[int(cls)]
                                print("label==============", label)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                                res.append(names[int(cls)])
                            print(names[int(cls)])
                # ============================检测条件逻辑编写区域============================================
                # if res != []:

                if actions_name_dict['action9'] in res and actions_name_dict['action10'] in res:
                    no_action_count = 0
                    print("吸烟人物位置信息：",person_head_info)
                    print("吸烟人手位置信息：",person_head_info)
                    #计算两个位置的锚框坐标位置信息 iou值：这里设置每隔10帧计算一下============
                    #这里要设置为如果连续2帧都有检测到有人在抽烟，就开始保存
                    two_action_count_list = []
                    #保存条件的开关
                    if action_smoke_save_key == False:
                        action_interval_count +=1
                        if action_interval_count % 3  == 0:
                            for head_i in person_head_info:
                                box1 = head_i
                                print("len(hand_info:::::::",len(hand_info))
                                for hand_i in hand_info:
                                    box2 = hand_i
                                    #调用iou计算函数
                                    iou_val =  count_iou(box1= box1,box2= box2)#计算锚框，，，，，
                                    print("每一次的iou值计算打印-----------",iou_val)
                                    if iou_val >0:
                                        two_action_count_list.append(iou_val)
                                        print("len------------lenaction_",len(two_action_count_list))
                            #如果列表是非空的就添加进去
                            if two_action_count_list is not None:
                                print("---------two_aciton_count_list:",len(two_action_count_list))
                                count_action_list.append(two_action_count_list)
                                two_action_count_list.clear()
                                print("-----------nnnulltwo_action_count_list:",two_action_count_list)
                            if len(count_action_list) >= 2:
                                action_smoke_save_key = True
                                count_action_list.clear()


                else:
                    no_action_count +=1
                    print("打印非头手动作计数帧数--------",no_action_count)
                    action_interval_count = 0

                #==========================================检测存储逻辑编写================================
                # 如果检测达到抽烟的条件就开始保存视频
                print("action_smoke_save_key================",action_smoke_save_key)
                if action_smoke_save_key == True:
                    action_smoke_save_key_count  += 1
                    print("action_smoke_save_key_count========", action_smoke_save_key_count )
                    if action_smoke_save_key_count  == 1:
                        now_time1 = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                        now_time = "{}-{}-{}-{}-{}-{}".format(now_time1[:4], now_time1[4:6], now_time1[6:8],
                                                              now_time1[8:10],
                                                              now_time1[10:12], now_time1[12:14])
                        ziduan_now_time = "{}-{}-{} {}:{}:{}".format(now_time1[:4], now_time1[4:6], now_time1[6:8],
                                                                     now_time1[8:10],
                                                                     now_time1[10:12], now_time1[12:14])
                        # 添加消息唯一编号id====这里选择毫秒级别
                        now_hm = datetime.datetime.now()
                        microseconds = now_hm.microsecond
                        now_hm = now_hm.strftime("%Y%m%d%H%M%S") + str(microseconds)
                        now_hm = str(now_hm)
                        MessageWarn_dict["algorithmId"] = int("04")
                        MessageWarn_dict["algorithmName"] = algor_dict["04"]["description"]
                        MessageWarn_dict["checkId"] = now_hm
                        MessageWarn_dict["checkTime"] = ziduan_now_time
                        MessageWarn_dict["checkResult"] = int("0")
                        MessageWarn_dict["alertTypeId"] = "041"
                        MessageWarn_dict["alertTypeName"] = algor_dict["04"]["041"]
                        MessageWarn_dict["alertMsg"] = algor_dict["04"]["041"]
                        #视频文件名称命名
                        save_path = actions_name_dict['equipment'] + "-" + actions_name_dict[
                            'action8'] + "-" + now_time + ".mp4"

                        video_save_path = get_video_save_path(save_path)
                        out_writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'avc1'), 25,
                                                     (im0.shape[1], im0.shape[0]))

                    print("存入")
                    # 在每一张图片上添加中文=======================================================================
                    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(im0)
                    # 选择一个支持中文字符的字体文件，例如：宋体（SimSun.ttf）
                    font_path = "/usr/share/fonts/truetype/dejavu/simsun.ttc"
                    font = ImageFont.truetype(font_path, 40)

                    draw = ImageDraw.Draw(im_pil)
                    # 定义新的时间戳============添加
                    text2 = "违规：存在人员吸烟"
                    position2 = (100, 160)
                    color = (0, 255, 0)

                    draw.text(position2, text2, font=font, fill=color)
                    # 将PIL图像格式转换回OpenCV图像格式
                    im0 = np.array(im_pil)
                    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                    # ==========================================================================================
                    img_list.append(im0)
                    # 开始写结束的条件
                    if action_smoke_save_key_count == save_count_capture:
                        #调动线程进行写入
                        img_list_b = img_list
                        cpm_MessageWarn_dict = MessageWarn_dict.copy()
                        cpm_MessageWarn_dict["deviceId"] = int(29)
                        thread1 = MyThread(front_frames_action=img_list_b, out_writer=out_writer,
                                           filepath=video_save_path, MessageWarn_dict=MessageWarn_dict,cpm_MessageWarn_dict=cpm_MessageWarn_dict)
                        thread1.start()
                        thread1.join()
                        print("线程调用完毕存入一个600")
                        img_list_b = []

                    elif ( no_action_count >= 250 and action_smoke_save_key_count < save_count_capture):
                        # 释放完资源之后将所有的计数器重置
                        cpm_MessageWarn_dict = MessageWarn_dict.copy()
                        cpm_MessageWarn_dict["deviceId"] = int(29)
                        thread1 = MyThread(front_frames_action=img_list, out_writer=out_writer,
                                           filepath=video_save_path, MessageWarn_dict=MessageWarn_dict,cpm_MessageWarn_dict=cpm_MessageWarn_dict)
                        thread1.start()
                        thread1.join()
                        print("线程调用完毕存入一个not200")
                        img_list_b = []
                        img_list = []
                        #参数重置=========================
                        action_interval_count = 0
                        action_smoke_save_key = False
                        action_smoke_save_key_count = 0
                        no_action_count = 0

                    elif ( no_action_count >= 250 and action_smoke_save_key_count > save_count_capture):
                        # 重置
                        action_interval_count = 0
                        action_smoke_save_key = False
                        action_smoke_save_key_count = 0
                        no_action_count = 0
                        img_list = []
                        img_list_b = []

                        #该段视频处理结束，重置参数=================


                # 获取当前时间===
                if action_smoke_save_key == 0:
                    t_now1 = time.time()
                    # 1800================
                    if int(t_now1) - int(t_now) == normal_sent_time:
                        print("正常截取===", t_now1 - t_now)
                        t_now = t_now1
                        # 调动线程开始发送一张图片
                        now_time1 = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                        now_time = "{}-{}-{}-{}-{}-{}".format(now_time1[:4], now_time1[4:6], now_time1[6:8],
                                                              now_time1[8:10],
                                                              now_time1[10:12], now_time1[12:14])
                        ziduan_now_time = "{}-{}-{} {}:{}:{}".format(now_time1[:4], now_time1[4:6], now_time1[6:8],
                                                                     now_time1[8:10],
                                                                     now_time1[10:12], now_time1[12:14])

                        save_path = actions_name_dict['equipment'] + "-" + actions_name_dict[
                            'action81'] + "-" + now_time + ".jpg"
                        video_save_path = get_video_save_path(save_path)
                        out_writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'avc1'), 25,
                                                     (im0.shape[1], im0.shape[0]))
                        # 添加消息唯一编号id====这里选择毫秒级别
                        now_hm = datetime.datetime.now()
                        microseconds = now_hm.microsecond
                        now_hm = now_hm.strftime("%Y%m%d%H%M%S") + str(microseconds)
                        now_hm = str(now_hm)

                        MessageWarn_dict["algorithmId"] = int("04")
                        MessageWarn_dict["algorithmName"] = algor_dict["04"]["description"]
                        MessageWarn_dict["checkId"] = now_hm
                        MessageWarn_dict["checkTime"] = ziduan_now_time
                        MessageWarn_dict["checkResult"] = int("1")

                        MessageWarn_dict["alertId"] = ""
                        MessageWarn_dict["alertTypeId"] = ""
                        MessageWarn_dict["alertTypeName"] = ""
                        MessageWarn_dict["alertMsg"] = ""
                        MessageWarn_dict["alertTime"] = ""

                        # 在每一张图片上添加中文==========================================
                        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                        im_pil = Image.fromarray(im0)
                        # 选择一个支持中文字符的字体文件，例如：宋体（SimSun.ttf）

                        font_path = "/usr/share/fonts/truetype/dejavu/simsun.ttc"
                        font = ImageFont.truetype(font_path, 40)

                        draw = ImageDraw.Draw(im_pil)
                        # 定义时间戳=============
                        put_now_time1 = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

                        # 绘制的字符
                        text = "定时发送"

                        position = (100, 100)
                        color = (0, 255, 0)

                        # 在PIL图像上绘制文本
                        draw.text(position, text, font=font, fill=color)

                        # 将PIL图像格式转换回OpenCV图像格式
                        im0 = np.array(im_pil)
                        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)

                        # =============================================
                        cpm_MessageWarn_dict = MessageWarn_dict.copy()
                        cpm_MessageWarn_dict["deviceId"] = int(29)
                        thread3 = MyThread(front_frames_action=im0, out_writer=out_writer, filepath=video_save_path,
                                           MessageWarn_dict=MessageWarn_dict,cpm_MessageWarn_dict=cpm_MessageWarn_dict)
                        thread3.start()
                        thread3.join()
                        # 写入完了之后需要重置参数，

                    # ========================================================================================
            t22 = time.time()
            print("t2时间差", t22 - t11)


if __name__ == '__main__':
    # #============定义告警消息json字典================================
    MessageWarn_dict = {
        "deviceId": "",
        "algorithmId": "",
        "algorithmName": "",
        "checkId": "",
        "checkTime": "",
        "checkResult": "",
        "alertId": "",
        "alertTypeId": "",
        "alertTypeName": "",
        "alertMsg": "",
        "alertTime": "",
        "relatePic": "",
        "relateVideo": "",
    }
    # #============================================
    # 获取视频流函数编写========================================================================================================
    device_id = str("50")
    son_device_id = str("29")
    if son_device_id == str("29"):
        MessageWarn_dict["deviceId"] = int(23)
    rtmp_object = RTMPLiveStream(device_id, son_device_id)
    token = rtmp_object.get_token_id()
    if token:
        rtmp = rtmp_object.get_rtmp()
        if rtmp:
            print("获取成功rtmp", rtmp)
        else:
            sys.exit()
    else:
        sys.exit()
    # # print(rtmp)
    global opt
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=r"../pytorch-yolov7-main/model_pt/0630xybest.pt",
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=rtmp,
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()


    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

