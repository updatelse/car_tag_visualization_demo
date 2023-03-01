#coding=utf-8
from distutils.file_util import move_file
import os
from tkinter import image_names
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt

img_suffix = ['jpg', 'png','JPG','PNG','jpeg','bmp','BMP']

def read_labels_txt(txt_path,image_path):      #实现读label.txt文件中的内容
    jpg_name = []
    car_coordinate = []
    label_files = os.listdir(txt_path)
    label_files.sort(key=lambda x:(x.split('.')[0]))                        #对读取的文件进行排序
    for x in label_files:
        single_lable = os.path.join(txt_path, x)
        for y in img_suffix:                  #找到与txt对应的图片名称
            img_file = "{}/{}.{}".format(image_path, x[:-4], y)
            if os.path.exists(img_file):
                jpg_name.append("{}.{}".format(x[:-4], y))
                break
        with open(single_lable, encoding='gbk') as f1:
            data = f1.readlines()
        for b in data:
            lab = b.split(" ")
            car_coordinate.append(lab[:-2])

    return jpg_name, car_coordinate
        

def dataloader(jpg_name,car_coordinate,image_path,num):  #实现每次loader出来9张

    num_stop = num + 9
    jpg_name_dataset = []
    for i in range(num, num_stop, 1):
        jpg_info = cv2.imread(os.path.join(image_path, jpg_name[i]))
        cv2.line(jpg_info, (int(car_coordinate[i][0]), int(car_coordinate[i][1])), (int(car_coordinate[i][2]), int(car_coordinate[i][3])), (0, 0, 255), 2)#依据坐标画四条直线
        cv2.line(jpg_info, (int(car_coordinate[i][2]), int(car_coordinate[i][3])), (int(car_coordinate[i][4]), int(car_coordinate[i][5])), (0, 0, 255), 2)
        cv2.line(jpg_info, (int(car_coordinate[i][4]), int(car_coordinate[i][5])), (int(car_coordinate[i][6]), int(car_coordinate[i][7])), (0, 0, 255), 2)
        cv2.line(jpg_info, (int(car_coordinate[i][6]), int(car_coordinate[i][7])), (int(car_coordinate[i][0]), int(car_coordinate[i][1])), (0, 0, 255), 2)
        # plt.imshow(jpg_info)
        # plt.show()
        crop_jpg_info = jpg_info[int(car_coordinate[i][1]) - 30:int(car_coordinate[i][5]) +30, int(car_coordinate[i][0]) - 30:int(car_coordinate[i][4]) + 30] #裁小图,往外扩了30个像素
        # plt.imshow(crop_jpg_info)
        # plt.show() 
        jpg_name_dataset.append([jpg_name[i],crop_jpg_info])
        
    return num_stop, jpg_name_dataset


def mouse_handler(event, x, y, flags, userdata):
    global x1, y1

    if event == cv2.EVENT_LBUTTONDOWN:   # 左键单击, 移走
        x1 = x // userdata[0][1] #w
        y1 = y // userdata[0][0] #h
        image_location = y1 * 3 + x1
        image_name = userdata[1][image_location][0]
        print("点击了一张图像,图像位置在第 %d 张, 名称为: %s, 已经移走 ✅ " % ((image_location + 1), image_name))
        print("------------------------------------------------------------")
        shutil.move(os.path.join(userdata[2] + "/"+ image_name), os.path.join(userdata[3] + "/picture/" + image_name))
        shutil.move(os.path.join(userdata[4] + "/"+ image_name[:-4]+ ".txt"), os.path.join(userdata[3] + "/labels/" + image_name[:-4]+ ".txt"))
        cv2.circle(userdata[5],(x,y), 15, (255,0,0), -1)
        cv2.imshow('multi', userdata[5])
        
    if event == cv2.EVENT_RBUTTONDBLCLK:    # 右键双击, 移回来, 即撤回
        x1 = x // userdata[0][1] #w
        y1 = y // userdata[0][0] #h
        image_location = y1 * 3 + x1
        image_name = userdata[1][image_location][0]
        print(" 正在执行撤回⚠️  将撤回您刚刚移走的那张图像,位置在第 %d 张, 名称为: %s, 已经移回来了" % ((image_location + 1), image_name))
        print("------------------------------------------------------------------------------------")
        shutil.move(os.path.join(userdata[3] + "/picture/" + image_name), os.path.join(userdata[2] + "/"+ image_name))
        shutil.move(os.path.join(userdata[3] + "/labels/" + image_name[:-4]+ ".txt"), os.path.join(userdata[4] + "/"+ image_name[:-4]+ ".txt"))
        cv2.circle(userdata[5],(x,y), 15, (240,32,160), -1)
        cv2.imshow('multi', userdata[5])
        

def show_multi_imgs(scale, imglist, order=None, border=0, border_color=(0, 0, 0)):
    """
    :param scale: float 原图缩放的尺度
    :param imglist: list 待显示的图像序列
    :param order: list or tuple 显示顺序 行×列
    :param border: int 图像间隔距离
    :param border_color: tuple 间隔区域颜色
    :return: 返回拼接好的numpy数组
    """
    if order is None:
        order = [1, len(imglist)]
    allimgs = imglist.copy()
    ws , hs = [], []
    for i, img in enumerate(allimgs):
        if np.ndim(img) == 2:
            allimgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        allimgs[i] = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        ws.append(allimgs[i].shape[1])
        hs.append(allimgs[i].shape[0])
    w = max(ws)
    h = max(hs)
    # 将待显示图片拼接起来
    sub = int(order[0] * order[1] - len(imglist))
    # 判断输入的显示格式与待显示图像数量的大小关系
    if sub > 0:
        for s in range(sub):
            allimgs.append(np.zeros_like(allimgs[0]))
    elif sub < 0:
        allimgs = allimgs[:sub]
    imgblank = np.zeros(((h+border) * order[0], (w+border) * order[1], 3)) + border_color
    imgblank = imgblank.astype(np.uint8)
    for i in range(order[0]):
        for j in range(order[1]):
            img_resize = cv2.resize(allimgs[i * order[1] + j],(w,h))
            imgblank[(i * h + i*border):((i + 1) * h+i*border), (j * w + j*border):((j + 1) * w + j*border), :] = img_resize
    return imgblank,h,w
 
 
if __name__ == '__main__':

#-------------------只要改下面这3行------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------------#  
    image_path = "F:/VScode/program/car_tag_demo/image9"             #放图片文件
    labels_txt_path = "F:/VScode/program/car_tag_demo/label9"        #放单个单个的label txt文件
    move_here = "F:/VScode/program/car_tag_demo/move_here"           #这个里面会建立2个文件夹, 一个picture文件夹, 一个labels文件夹,跟mouse_handler函数最后2行有关联.
    num = 0
#-----------------------------------------------------------------------------------------------------------------------------------------------#   
    

    if not os.path.exists(move_here + "/" + "picture"):
        os.makedirs(move_here + "/" + "picture") 
    if not os.path.exists(move_here + "/" + "labels"):
        os.makedirs(move_here + "/" + "labels")  
    jpg_name, car_coordinate = read_labels_txt(labels_txt_path, image_path)
    for x in range(len(jpg_name)//9):
        datasets = []
        p = []
        nums, p = dataloader(jpg_name, car_coordinate, image_path, num)
        num = nums
        img,h,w = show_multi_imgs(1, [p[0][1],p[1][1],p[2][1],p[3][1],p[4][1],p[5][1],p[6][1],p[7][1],p[8][1]], (3, 3))
        datasets.append([h,w])
        datasets.append(p)
        datasets.append(image_path)
        datasets.append(move_here)
        datasets.append(labels_txt_path)
        datasets.append(img)
        cv2.namedWindow('multi') 
        cv2.setMouseCallback("multi", mouse_handler, datasets)
        cv2.imshow('multi', img)
        
        k = cv2.waitKey(0) & 0xFF
        if k == 99:                               #按q键 加载入下一批9张
            cv2.destroyAllWindows()
    print("您本次共浏览了 %d 批(每批9张)车牌" % (num // 9))
    count_images = os.listdir(os.path.join(move_here,"picture"))
    print("您总共移走了 %d 张不规范车牌" % len(count_images))
    print("下次再工作时,请您把num修改成 %d " % (((num // 9) - len(count_images) // 9) * 9))