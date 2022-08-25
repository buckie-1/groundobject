from sklearn.model_selection import train_test_split
from sklearn import ensemble
import os
from sklearn import metrics
from imblearn.over_sampling import SMOTE  # 随机采样函数 和SMOTE过采样函数
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import recall_score


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


class DealDataset(Dataset):

    def __init__(self, train_x, train_y):
        self.x_data, self.y_data = self.get_data(train_x, train_y)
        self.len = len(self.x_data)

    def get_data(self, train_x, train_y):

        data_x_normal = self.normalization(train_x)
        data_y_normal = []
        for i in train_y:
            # if i[0] == 0:
            #     data_y_normal.append(0)
            # if i[0] == 1:
            #     data_y_normal.append(1)
            data_y_normal.append(int(i))

        return data_x_normal, data_y_normal

    def normalization(self, data):
        scaler = MinMaxScaler()
        data_normal = scaler.fit_transform(data)
        return data_normal

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


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


def main():
    warnings.filterwarnings("ignore")

    net = Net2(input_dim=4, output_dim=2, hidden_dim=1000)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    lr = 0.001
    Epoch = 500
    best_acc = 0.0
    best_recall = 0.0
    best_epoch = 0
    history = []
    best_cm = []
    best_ma_f1 = 0
    best_mi_f1 = 0
    save_path = r'E:\作物分类\model'  # 模型保存地址
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # weight=torch.from_numpy(np.array([0.9,0.1])).float()
    loss_func = nn.CrossEntropyLoss()
    #     loss_func = focal_loss(alpha=0.6, gamma=3, num_classes=2, size_average=True)
    mult_step_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.8)

    train_loss = []
    valid_loss = []
    train_steps = len(train_data)
    train_data_size = len(train_data)
    valid_data_size = len(test_data)

    net.to(device)
    for i, epoch in enumerate(range(Epoch)):
        total_train_loss = []
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        train_recall = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        valid_recall = 0.0
        ma_f1 = 0.0
        mi_f1 = 0.0
        cm = np.zeros((2, 2))
        for j, data in enumerate(train_data):
            x, y = data
            # x, y = Variable(x).float(), Variable(y).float()
            x, y = torch.FloatTensor(x.float()).to(
                device), torch.FloatTensor(y.float()).to(device)
            prediction = net(x)
            loss = loss_func(prediction, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            ret, predictions = torch.max(prediction.data, 1)
            labels = y.to(device)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item()

            recall = recall_score(predictions.detach().numpy(), y.detach().numpy(), pos_label=0)
            train_recall += recall

        ################################# eval ###############################

        acc = 0.0
        net.eval()
        for step, (b_x, b_y) in enumerate(test_data):
            b_x = torch.FloatTensor(b_x.float()).to(device)
            b_y = torch.FloatTensor(b_y.float()).to(device)
            pred = net(b_x)
            loss = loss_func(pred, b_y.long())
            valid_loss += loss.item()

            y_pred = np.argmax(pred.detach().numpy(), axis=1)

            ret, predictions = torch.max(pred.data, 1)
            val_labels = b_y.to(device)
            correct_counts = predictions.eq(val_labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            valid_acc += acc.item()

            recall = recall_score(predictions.detach().numpy(), b_y.detach().numpy(), pos_label=0)
            valid_recall += recall

            ma_f1 = f1_score(val_labels.cpu(), predictions.cpu(), average='macro')
            mi_f1 = f1_score(val_labels.cpu(), predictions.cpu(), average='micro')

            cm0 = confusion_matrix(b_y.detach().numpy(), y_pred, labels=None, sample_weight=None)
            cm += cm0
            # total_valid_loss[step] = loss.item()
        # valid_loss.append(np.mean(total_valid_loss))
        # valid_loss.append(torch.mean(total_valid_loss).detach().cpu())

        lr = optimizer.param_groups[0]['lr']

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_train_recall = train_recall / train_data_size

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size
        avg_valid_recall = valid_recall / valid_data_size

        avg_ma_f1 = ma_f1 / valid_data_size
        avg_mi_f1 = mi_f1 / valid_data_size

        history.append(
            [avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc, avg_train_recall, avg_valid_recall])

        if avg_valid_acc > best_acc:
            best_acc = avg_valid_acc
            best_recall = avg_valid_recall
            best_epoch = epoch + 1
            best_cm = cm
            best_ma_f1 = avg_ma_f1
            best_mi_f1 = avg_mi_f1
            print(best_acc)
            if best_acc > 0.95:
                model_path = save_path + r'\network' + str(best_acc) + '.pth'
                print(model_path)
                torch.save(net.state_dict(), model_path)
                print('save!')

        mult_step_scheduler.step()

        print(str(datetime.datetime.now()) + ': ')
        print('[epoch %d] train_loss: %.4f  train_accuracy: %.4f  train_recall: %.4f'
              '  val_loss: %.4f  val_accuracy: %.4f  val_recall: %.4f  lr:%.8f' %
            (i + 1, avg_train_loss, avg_train_acc, avg_train_recall, avg_valid_loss, avg_valid_acc, avg_valid_recall,
             lr))

    df_cm = pd.DataFrame(best_cm)
    plt.figure(figsize=(8, 8))
    sns.heatmap(df_cm, annot=True, fmt='')
    title = 'Multi-Class Confusion Matrix'
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('%s.jpg' % title, bbox_inches='tight')
    plt.show()

    print('Finished')
    return best_epoch, best_acc, best_ma_f1, best_mi_f1


if __name__ == '__main__':
    data_path = r'E:\作物分类\data' # 数据地址
    # land_path = r'E:\作物分类\jgvqzuqqejdctfpx'
    map_index = ['sxh-data', 'syg-data', 'syh-data']
    # land_index = ['sxh-land', 'syg-land', 'syh-land']
    map_x, map_y = read_data(data_path, map_index)
    # land_x, land_y = read_data(land_path, land_index)
    # data_x = np.vstack((map_x, land_x))
    # data_y = np.vstack((map_y, land_y))
    # 打乱数据
    index = [i for i in range(len(map_x))]
    np.random.shuffle(index)
    data_x = map_x[index]
    data_y = map_y[index]

    smote = SMOTE(random_state=0)  # random_state为0（此数字没有特殊含义，可以换成其他数字）使得每次代码运行的结果保持一致
    X_smotesampled, y_smotesampled = smote.fit_resample(data_x, data_y)  # 使用原始数据的特征变量和目标变量生成过采样数据集
    print('Smote法过采样后', Counter(y_smotesampled))

    dataset = DealDataset(X_smotesampled, y_smotesampled)
    length = [int(len(dataset) * 0.8)]
    length.append(len(dataset) - length[0])
    train_db, test_db = torch.utils.data.random_split(dataset, [int(length[0]), int(length[1])])
    train_data = DataLoader(dataset=train_db, batch_size=400, shuffle=True)
    test_data = DataLoader(dataset=test_db, batch_size=200, shuffle=False)

    best_epoch, best_acc, best_ma_f1, best_mi_f1 = main()
    print('best_epoch:%d , best_acc:%.4f, best_ma_f1:%.4f, best_mi_f1:%.4f'
          % (best_epoch, best_acc, best_ma_f1, best_mi_f1))

