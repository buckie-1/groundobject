import gdal
from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
import csv
import pickle
import torch.nn as nn
import torch
import torch.nn.functional as F
from osgeo import ogr, osr


class Net2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Net2, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        y = self.fc3(x) + x
        y = F.relu(y)
        y = self.fc4(y)
        y = F.relu(y)
        y = self.fc4(y)
        y = F.relu(y)
        z = self.fc4(y) + y
        z = F.relu(z)
        z = self.fc4(z)
        z = F.relu(z)
        z = self.fc4(z)
        z = F.relu(z)
        m = self.fc4(z) + z
        m = F.relu(m)
        m = self.fc4(m)
        m = F.relu(m)
        m = self.fc4(m)
        m = F.relu(m)
        m = self.fc4(m)
        m = F.relu(m)
        n = self.fc4(m) + m
        n = F.relu(n)
        n = self.fc5(n)
        return n


def read_data(filename, map):
    x, y = [], []
    for i in map:
        path = os.path.join(filename, i + '.csv')
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        x.append(data[:, :4])
        y.append(data[:, 4])
    data_x = np.vstack((np.vstack((x[0], x[1])), x[2]))
    data_y = np.vstack((np.vstack((y[0].reshape(-1, 1), y[1].reshape(-1, 1))), y[2].reshape(-1, 1)))
    return data_x, data_y


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
        labels.append(int(str[i][2]))
    return x_min, x_max, y_min, y_max, labels


def getSRSPair(dataset):
    '''
    ??????????????????????????????????????????????????????
    :param dataset: GDAL????????????
    :return: ?????????????????????????????????
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs


def get_points(mapname, imgname, x, y):
    root_path = r'D:\arcgis\output\deeplearning'
    V_index = ['ndvi', 'ndwi', 'rvi', 'savi']
    points = []
    for i in V_index:
        map_path = mapname + '-' + i
        path = os.path.join(root_path, map_path)
        img_path = os.path.join(path, imgname)
        img = Image.open(img_path)
        imgs = np.array(img)
        points.append(imgs[x, y])
    points = np.array(points)
    points.reshape(1, -1)
    return points


if __name__ == '__main__':
    ## ????????????????????? ##
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.CreateDataSource("sxh.shp")  ## shp????????????
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  ## ???????????????WGS84
    layer = data_source.CreateLayer("sxh", srs, ogr.wkbPoint)  ## ??????????????????shp????????????
    field_name = ogr.FieldDefn("type", ogr.OFTString)  # ????????????
    field_name.SetWidth(20)  # ????????????
    layer.CreateField(field_name)  # ????????????

    # ????????????
    model = Net2(input_dim=4, output_dim=2, hidden_dim=1000)
    model.load_state_dict(torch.load(r'E:\????????????\model\modelnetwork0.9541449735148392.pth'))
    model.eval()

    root_path = r'D:\arcgis\output\deeplearning'
    map_index = ['sxh', 'syg', 'syh']
    V_index = ['ndvi', 'ndwi', 'rvi', 'savi']
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    ypre = []

    for i in map_index:
        map_path = []
        for ii in V_index:
            V_name = i + '-' + ii
            map_path.append(os.path.join(root_path, V_name))  # ??????map??????
        img_dir, lab_dir = readmap(map_path[0])  # ??????img???label?????????
        print('star drawing {}'.format(i))
        for j in range(1, len(img_dir)):
            img_path = os.path.join(map_path[0], img_dir[j])
            lab_path = os.path.join(map_path[0], lab_dir[j])
            x_min, x_max, y_min, y_max, labels = getlabel(lab_path)
            dataset = gdal.Open(img_path)  # ??????tif
            adfGeoTransform = dataset.GetGeoTransform()  # ??????????????????
            for l in range(len(x_min)):
                for y in range(y_min[l], y_max[l]):
                    for x in range(x_min[l], x_max[l]):
                        px = adfGeoTransform[0] + x * adfGeoTransform[1] + y * adfGeoTransform[2]
                        py = adfGeoTransform[3] + x * adfGeoTransform[4] + y * adfGeoTransform[5]
                        prosrs, geosrs = getSRSPair(dataset)
                        ct = osr.CoordinateTransformation(prosrs, geosrs)
                        coords = ct.TransformPoint(px, py)
                        coord = coords[:2]  # ???????????????????????????
                        points = get_points(i, img_dir[j], x, y)  # ????????????????????????
                        points = scaler.fit_transform(points.reshape(1, -1))  # ?????????
                        pred = model(points)
                        y_pred = np.argmax(pred.detach().numpy(), axis=1)  # ????????????????????????
                        ypre.append(y_pred[0])

                        # ??????shp???
                        feature = ogr.Feature(layer.GetLayerDefn())
                        if y_pred[0] == 0:
                            feature.SetField("type", '0')
                        if y_pred[0] == 1:
                            feature.SetField("type", '1')  # ???????????????
                        wkt = "POINT(%f %f)" % (float(coord[1]), float(coord[0]))  # ?????????
                        point = ogr.CreateGeometryFromWkt(wkt)  # ?????????
                        feature.SetGeometry(point)  # ?????????
                        layer.CreateFeature(feature)  # ?????????
            print('{} drawing finished'.format(img_dir[j]))
        #     break
        # break

        print('map {} drawing finished'.format(i))
        break

    feature = None  ## ????????????
    data_source = None  ## ????????????

    print('finished')

