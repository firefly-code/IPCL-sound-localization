from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
import csv
from sklearn import preprocessing
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from pathlib import Path
from PIL import Image


import re


def spherePlot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, np.pi, 72)

    # create the sphere surface
    x=1.5 * np.outer(np.cos(u), np.sin(v))
    y=1.5 * np.outer(np.sin(u), np.sin(v))
    z=1.5 * np.outer(np.ones(np.size(u)), np.cos(v))

    #grid[labels]= probility
    myheatmap = data * 3
    ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cm.PiYG(myheatmap),shade=False)
    plt.show()

def get_azi_ele_value(data):
    changed = []
    azi_ele= np.ones((72,72))*-1
    for label,pred in data:
        with open('labels.csv', 'rt') as f:
                reader = csv.reader(f, delimiter=',') 
                for row in reader:
                    if(int(row[2])==label):
                        # print("lab2:",row[0],row[1])
                        azi = (int(int(row[0])/5))
                        ele = (int((180+int(row[1]))/5))
                        # print("lab:",row[2])
                        # print("azi:",azi)
                        # print("ele:",ele)
                        if(azi_ele[azi,ele] != -1):
                            azi_ele[azi,ele]= (azi_ele[azi,ele]+(label-pred))/2
                        else:
                            azi_ele[azi,ele]= (label-pred)

    points = np.array(azi_ele) 
    indexs = np.where(points == -1)
    points[indexs]= 0
    # norm = gaussian_filter(points, sigma=2)
    # norm=preprocessing.normalize(norm)
    # norm[norm == 0] = 1
    plt.imshow(points)
    return points

def confusion_matrix_graph(data):
    cm = confusion_matrix(data[:,0],data[:,1],normalize='all')
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=False, fmt='d', cmap='BuGn', square=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(190),labels=np.arange(190), rotation=90,fontsize=5)
    plt.yticks(np.arange(190),labels=np.arange(190), rotation=0, fontsize=5)
    plt.show()


def confusion_matrix_graph_elevations(data):
    data_ele = defaultdict(list)
    data_azi = defaultdict(list)
    with open('labels.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',') 
            for row in reader:
                data_ele[row[1]].append(int(row[2]))
                data_azi[int(row[2])].append(int(row[0]))
                
                
    for key, lab_ele in data_ele.items():
        indexs = np.where(np.in1d(data[:,0],lab_ele))
        x= [data_azi[ele]for ele in data[indexs,0].squeeze()]
        y= [data_azi[ele]for ele in data[indexs,1].squeeze()]
        print(data_azi)
        cm = confusion_matrix(x,y,normalize='all')
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=False, fmt='d', cmap='cividis', square=True)
        plt.title(f'Confusion Matrix {key}',fontsize= 30)
        plt.xlabel('Predicted',fontsize= 30)
        plt.ylabel('True',fontsize= 30)
        plt.xticks(np.arange(19),labels=[270,280,290,300,310,320,330,340,350,0,10,20,30,40,50,60,70,80,90,], rotation=90,fontsize= 20)
        plt.yticks(np.arange(19),labels=[270,280,290,300,310,320,330,340,350,0,10,20,30,40,50,60,70,80,90,], rotation=0,fontsize= 20)
        plt.savefig(f'Data/ImagesOfGraphs/confusion_matrix_{key}.png')
        
        
def combine_confusion_matrix():
    image_files = ['confusion_matrix_-45.png', 'confusion_matrix_-30.png', 'confusion_matrix_-20.png', 'confusion_matrix_-10.png', 'confusion_matrix_0.png', 'confusion_matrix_10.png', 'confusion_matrix_20.png', 'confusion_matrix_30.png', 'confusion_matrix_45.png', 'confusion_matrix_60.png']
    images = [Image.open(f"Data/ImagesOfGraphs/{filename}") for filename in image_files]
    image_width, image_height = images[0].size
    combined_image = Image.new('RGB', (image_width * 5, image_height * 2))
    
    for i, img in enumerate(images):
        row = i % 2
        col = i // 2
        combined_image.paste(img, (col * image_width, row * image_height))

    combined_image.save('Data/ImagesOfGraphs/combined_image.png')
def general_x_y_graphs(data_path,title,yaxis,xaxis,smoothing):
    csv_file_path = data_path
    data = pd.read_csv(csv_file_path)
    steps = data['Step']
    values = data['Value']
    window_size = smoothing 
    smoothed_values = values.rolling(window=window_size).mean()
 
    plt.figure(figsize=(10, 6))
    plt.plot(steps, smoothed_values, color='green', linewidth=2, alpha=1, label='Smoothed Data')
    plt.plot(steps, values, alpha=0.5, color ='green')
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.title(title)
    plt.grid(True)
    plt.show()


def train_val_accuracy_comparison(data_path_train,data_path_val,title,yaxis,xaxis):
    
    data_train = pd.read_csv(data_path_train)
    data_val = pd.read_csv(data_path_val)


    steps = data_train['Step']
    values_train = data_train['Value']
    values_val =data_val['Value']
    
    window_size = 4
    smoothed_values_train = values_train.rolling(window=window_size).mean()
    smoothed_values_val = values_val.rolling(window=window_size).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, smoothed_values_train, color='green', linewidth=2, alpha=1)
    plt.plot(steps, values_train, alpha=0.2, color ='green',label='Data Training')
    plt.plot(steps, smoothed_values_val, color='purple', linewidth=2, alpha=1)
    plt.plot(steps, values_val, alpha=0.2, color ='purple',label='Data Validation')
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    

def classes_distribution(data):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=190, edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Numbers')
    plt.grid(True)

def classes_distribution_Dataset(data):
    all_labels=[]
    for path in list(Path(data).glob('*.npz')):
        pattern = r"Az_(?P<azimuth>-?\d+)_El_(?P<elevation>-?\d+)"
        labelData = re.search(pattern, str(path))
        azimuth = int(labelData.group('azimuth'))
        elevation = int(labelData.group('elevation'))
        label = 0
        with open('labels.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',') 
            for row in reader:
                if(int(row[0])==azimuth and int(row[1])==elevation):
                    label = row[2]
                    break
        all_labels.append(label)
    
    plt.figure(figsize=(50, 6))
    plt.hist(all_labels, bins=190, edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Numbers')
    plt.grid(True)

def correctly_labeled(data):
    count = 0
    for i,ele in enumerate(data):
        print(ele[1],ele[0])
        if ele[1]==ele[0]:
            count = count+1
    print(count/i)
    


data = np.load("val_data5.npy").squeeze()
# correctly_labeled(data)
# print(data.shape)
# print(data[:,0].shape)
# print(data[:,1].shape)
#classes_distribution_Dataset("Data/Train")
# plt.show()
# print(data[:,1].shape)
# classes_distribution(data[:,1])
# classes_distribution(data[:,0])

# print(data.shape)
# confusion_matrix_graph_elevations(data)
# combine_confusion_matrix()
# loss_train = "Data/ResultsCSV/run-May23_18-10-50_ValkBig-tag-Loss_train.csv"
# acc_train = "Data/ResultsCSV/run-May23_18-10-50_ValkBig-tag-Acc_train.csv"
# acc_val = "Data/ResultsCSV/run-May23_18-10-50_ValkBig-tag-Acc_Val.csv"
# general_x_y_graphs(loss_train,"Loss training","Loss","Step",20)
# general_x_y_graphs(acc_train,"Accuracy training","Acc","Step",20)
# general_x_y_graphs(acc_val,"Accuracy validation","Acc","Epoch",4)
# loss_train ="Data/ResultsCSV/run-resnet_ssl20240523_181050_Training vs. Validation Loss_Training-tag-Training vs. Validation Loss.csv"
# loss_val = "Data/ResultsCSV/run-resnet_ssl20240523_181050_Training vs. Validation Loss_Validation-tag-Training vs. Validation Loss.csv"
# train_val_accuracy_comparison(loss_train,loss_val,"Train vs Validation Loss","Loss","EPOCH")
# plt.show()
data= get_azi_ele_value(data)
spherePlot(data)
plt.show()


