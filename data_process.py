from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
import csv



class DealDataset(Dataset):
    """


    """

    def __init__(self, train_x, train_y):
        self.x_data, self.y_data = self.get_data(train_x, train_y)
        self.len = len(self.x_data)

    def get_data(self, train_x, train_y):

        data_x_normal = self.normalization(train_x)
        data_y_normal = train_y

        return data_x_normal, data_y_normal

    def normalization(self, data):
        scaler = MinMaxScaler()
        data_normal = scaler.fit_transform(data)
        return data_normal

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def readmap(filename):
    file_patn = os.path.join(filename, 'map.txt')
    f = open(file_patn, 'r', encoding='UTF-8')
    str = f.readlines()
    img_dir, lab_dir = [], []
    for i in range(len(str)):
        img_dir.append(str[i][0:20])
        lab_dir.append(str[i][22:42])
    return img_dir, lab_dir


def getlabel(filename):
    f = open(filename, 'r', encoding='UTF-8')
    str = f.readlines()
    x_min, x_max, y_min, y_max, labels = [], [], [], [], []
    for i in range(len(str)):
        x_min.append(int(float(str[i][14:17])))
        x_max.append(int(float(str[i][30:33])))
        y_min.append(int(float(str[i][22:25])))
        y_max.append(int(float(str[i][38:41])))
        # 只区分玉米和其他作物
        # if str[i][2] == '1':
        #     labels.append(1)
        # elif str[i][2] == '0':
        #     labels.append(0)
        # else:
        #     labels.append(2)
        labels.append(int(str[i][2]))
    return x_min, x_max, y_min, y_max, labels


def getdata(filename, x_min, x_max, y_min, y_max, labels):
    img = Image.open(filename)
    imgs = np.array(img)
    all_data = []
    for i in range(len(labels)):
        xmin = x_min[i] + int((x_max[i] - x_min[i]) * 0.4)
        xmax = x_min[i] + int((x_max[i] - x_min[i]) * 0.6)
        ymin = y_min[i] + int((y_max[i] - y_min[i]) * 0.4)
        ymax = y_min[i] + int((y_max[i] - y_min[i]) * 0.6)
        points = np.reshape(imgs[xmin:xmax, ymin:ymax], (-1, 1))
        label = np.full((points.shape[0], 1), int(labels[i]))
        all_data.append(np.hstack((points, label)))
    data = all_data[0]
    for i in range(1, len(all_data)):
        data = np.vstack((data, all_data[i]))
    return data


if __name__ == '__main__':
    # 将图片像素点保存在本地
    path1 = r'D:\arcgis\output\deeplearning'
    path2 = r'E:\作物分类\jgvqzuqqejdctfpx'
    VIndex = [['sxh-ndvi', 'sxh-ndwi', 'sxh-rvi', 'sxh-savi'], ['syg-ndvi', 'syg-ndwi', 'syg-rvi', 'syg-savi'],
              ['syh-ndvi', 'syh-ndwi', 'syh-rvi', 'syh-savi']]
    datas = []
    for i, name in enumerate(VIndex):
        for k in name:
            points = []
            p = []
            root_dir = os.path.join(path1, k)
            img_dir, lab_dir = readmap(root_dir)
            for j in range(1, len(img_dir)):
                img_path = os.path.join(root_dir, img_dir[j])
                lab_path = os.path.join(root_dir, lab_dir[j])
                x_min, x_max, y_min, y_max, labels = getlabel(lab_path)
                data = getdata(img_path, x_min, x_max, y_min, y_max, labels)
                points.append(data)
            p = points[0]
            for m in range(1, len(points)):
                p = np.vstack((p, points[m]))
            datapath = k + 'data.csv'
            landpath = k + 'land.csv'
            with open(os.path.join(r'E:\作物分类\data2', datapath), 'w', newline='') as file:
                mywriter = csv.writer(file, delimiter=',')
                mywriter.writerows(p)
            print('{} data collecting finished'.format(k))





