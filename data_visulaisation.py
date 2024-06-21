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
import torch
import os

import re

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def spherePlot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, np.pi, 72)
    
    x=1.5 * np.outer(np.cos(u), np.sin(v))
    y=1.5 * np.outer(np.sin(u), np.sin(v))
    z=1.5 * np.outer(np.ones(np.size(u)), np.cos(v))

    #grid[labels]= probility
    myheatmap = data * 3
    ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cm.PiYG(myheatmap),shade=False)
    plt.show()
    
def spherePlotConfusionElevation(data):
    azis=[270,280,290,300,310,320,330,340,350,0,10,20,30,40,50,60,70,80,90]
    azi_ele= np.ones((72,72))*-1
    data_ele = defaultdict(list)
    data_azi = defaultdict(list)
    with open('labels.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',') 
            for row in reader:
                data_ele[row[1]].append(int(row[2]))
                data_azi[int(row[2])].append(int(row[0]))

    for key, lab_ele in data_ele.items():
        indexs = np.where(np.in1d(data[:,0],lab_ele))
        x=[data_azi[ele] for ele in data[indexs,0].squeeze()]
        y=[data_azi[ele] for ele in data[indexs,1].squeeze()]
        conmat = confusion_matrix(x,y,normalize='true')
        for i in range(19):
            azi = (int(azis[i]/5))
            ele = (int((180+int(key))/5))
            # print("lab:",row[2])
            # print("azi:",azi)
            # print("ele:",ele)
            azi_ele[azi,ele]=conmat[i,i]

        
    norm = azi_ele
    norm = gaussian_filter(azi_ele,sigma=0.1)
    #norm=preprocessing.normalize(norm)
    plt.imshow(azi_ele)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, np.pi, 72)

    x=1.5 * np.outer(np.cos(u), np.sin(v))
    y=1.5 * np.outer(np.sin(u), np.sin(v))
    z=1.5 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    print(norm.shape)
    #grid[labels]= probility
    myheatmap = norm
    new_cmap = cm.RdYlGn
    new_cmap.set_under("grey")
    ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=new_cmap(myheatmap),shade=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
def spherePlotConfusionAzimuth(data):
    eles=[-45,-30,-20,-10,0,10,20,30,45,60]
    azi_ele= np.ones((72,72))*-1
    data_ele = defaultdict(list)
    data_azi = defaultdict(list)
    with open('labels.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',') 
            for row in reader:
                data_azi[row[0]].append(int(row[2]))
                data_ele[int(row[2])].append(int(row[1]))

    for key, lab_azi in data_azi.items():
        indexs = np.where(np.in1d(data[:,0],lab_azi))
        x= [data_ele[azi] for azi in data[indexs,0].squeeze()]
        y= [data_ele[azi] for azi in data[indexs,1].squeeze()]
        conmat = confusion_matrix(x,y,normalize='true')
        for i in range(10):
            azi = (int(int(key)/5))
            ele = (int((180+int(eles[i]))/5))
            # print("lab:",row[2])
            # print("azi:",azi)
            # print("ele:",ele)
            azi_ele[azi,ele]=conmat[i,i]

        
    norm = azi_ele
    norm = gaussian_filter(azi_ele,sigma=0.1)
    #norm=preprocessing.normalize(norm)
    plt.imshow(azi_ele)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, np.pi, 72)


    x=1.5 * np.outer(np.cos(u), np.sin(v))
    y=1.5 * np.outer(np.sin(u), np.sin(v))
    z=1.5 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    print(norm.shape)
    #grid[labels]= probility

    myheatmap = norm
    new_cmap = cm.RdYlGn
    new_cmap.set_under("grey")
    ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=new_cmap(myheatmap),shade=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

def get_azi_ele_value(data):
    changed = []
    azis = []
    eles = []
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
                        print(azi,ele)
                        azis.append(azi)
                        eles.append(ele)
                        if(azi_ele[azi,ele] != -1):
                            azi_ele[azi,ele]= (azi_ele[azi,ele]+(label-pred))/2
                        else:
                            azi_ele[azi,ele]= (label-pred)

    points = np.array(azi_ele) 
    indexs = np.where(points == -1)
    points[indexs]= 0
    norm = gaussian_filter(points, sigma=2)
    norm=preprocessing.normalize(norm)
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
        sns.heatmap(cm, annot=False, fmt='d', cmap='cividis', square=True,vmin=0,vmax=0.06)
        sns.set(font_scale=2)
        plt.title(f'Elevation {key}°',fontsize= 40)
        plt.xlabel('Predicted',fontsize= 40)
        plt.ylabel('True',fontsize= 40)
        plt.xticks(np.arange(19),labels=[270,280,290,300,310,320,330,340,350,0,10,20,30,40,50,60,70,80,90,], rotation=90,fontsize= 30)
        plt.yticks(np.arange(19),labels=[270,280,290,300,310,320,330,340,350,0,10,20,30,40,50,60,70,80,90,], rotation=0,fontsize= 30)
        plt.savefig(f'Data/ImagesOfResultGraphs/ConfusionElevationResnet34/confusion_matrix_ele_{key}.png')

def confusion_matrix_graph_azimuth(data):
    data_ele = defaultdict(list)
    data_azi = defaultdict(list)
    with open('labels.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',') 
            for row in reader:
                data_azi[row[0]].append(int(row[2]))
                data_ele[int(row[2])].append(int(row[1]))
                
    for key, lab_azi in data_azi.items():
        print(lab_azi)
        print(data[:,0])
        indexs = np.where(np.in1d(data[:,0],lab_azi))
        x= [data_ele[azi] for azi in data[indexs,0].squeeze()]
        y= [data_ele[azi] for azi in data[indexs,1].squeeze()]
        cm = confusion_matrix(x,y,normalize='all')
        plt.figure(figsize=(15, 15)) 
        sns.heatmap(cm, annot=False, fmt='d', cmap='cividis', square=True,vmin=0,vmax=0.12)
        sns.set(font_scale=2)
        plt.title(f'Azimuth {key}°',fontsize= 40)
        plt.xlabel('Predicted',fontsize= 40)
        plt.ylabel('True',fontsize= 40)
        plt.xticks(np.arange(10),labels=[-45,-30,-20,-10,0,10,20,30,45,60], rotation=90,fontsize= 30)
        plt.yticks(np.arange(10),labels=[-45,-30,-20,-10,0,10,20,30,45,60], rotation=0,fontsize= 30)
        plt.savefig(f'Data/ImagesOfResultGraphs/ConfusionAzimuthResnet34/confusion_matrix_azi_{key}.png')
        
# image_files = ['confusion_matrix_ele_60.png', 'confusion_matrix_ele_45.png', 'confusion_matrix_ele_30.png', 'confusion_matrix_ele_20.png', 'confusion_matrix_ele_10.png', 'confusion_matrix_ele_0.png', 'confusion_matrix_ele_-10.png', 'confusion_matrix_ele_-20.png', 'confusion_matrix_ele_-30.png', 'confusion_matrix_ele_-45.png']
# image_files = ['confusion_matrix_azi_270.png', 'confusion_matrix_azi_280.png', 'confusion_matrix_azi_290.png', 'confusion_matrix_azi_300.png', 'confusion_matrix_azi_310.png', 'confusion_matrix_azi_320.png', 'confusion_matrix_azi_330.png', 'confusion_matrix_azi_340.png', 'confusion_matrix_azi_350.png', 'confusion_matrix_azi_0.png','confusion_matrix_azi_0.png', 'confusion_matrix_azi_10.png', 'confusion_matrix_azi_20.png', 'confusion_matrix_azi_30.png', 'confusion_matrix_azi_40.png', 'confusion_matrix_azi_50.png', 'confusion_matrix_azi_60.png', 'confusion_matrix_azi_70.png', 'confusion_matrix_azi_80.png', 'confusion_matrix_azi_90.png']
       
        
        
def combine_confusion_matrix_elevations(data):
    image_files = ['confusion_matrix_ele_10.png', 'confusion_matrix_ele_60.png', 'confusion_matrix_ele_0.png', 'confusion_matrix_ele_45.png', 'confusion_matrix_ele_-10.png', 'confusion_matrix_ele_-45.png']
    images = [Image.open(f"{data}/{filename}") for filename in image_files]
    print(len(images))
    image_width, image_height = images[0].size
    combined_image = Image.new('RGB', (image_width * 3, image_height * 2))
    
    for i, img in enumerate(images):
        row = i % 2
        col = i // 2
        combined_image.paste(img, (col * image_width, row * image_height))

    combined_image.save(f'Data/ImagesOfResultGraphs/combined_image_{os.path.basename(data)+"Only3"}.png')

def combine_confusion_matrix_azimuth(data):
    image_files = ['confusion_matrix_azi_270.png', 'confusion_matrix_azi_0.png', 'confusion_matrix_azi_90.png']
    images = [Image.open(f"{data}/{filename}") for filename in image_files]
    print(len(images))
    image_width, image_height = images[0].size
    combined_image = Image.new('RGB', (image_width * 3, image_height * 1))
    
    for i, img in enumerate(images):
        row = 0
        col = i 
        combined_image.paste(img, (col * image_width, row * image_height))

    combined_image.save(f'Data/ImagesOfResultGraphs/combined_image_{os.path.basename(data)+"Only3"}.png')
    
def general_x_y_graphs(data_path_34,data_path_18,title,yaxis,xaxis,smoothing):
    csv_file_path = (data_path_34,data_path_18)
    
    data34 = pd.read_csv(csv_file_path[0])
    data18 = pd.read_csv(csv_file_path[1])
    steps34 = data34['Step']
    values34 = data34['Value']
    steps18 = data18['Step']
    values18 = data18['Value']
    window_size = smoothing 
    smoothed_values34 = values34.rolling(window=window_size).mean()
    smoothed_values18 = values18.rolling(window=window_size).mean()
 
    plt.figure(figsize=(10, 6))
    plt.plot(steps34, smoothed_values34*100, color='green', linewidth=2, alpha=1, label='Average')
    plt.plot(steps34, values34*100, alpha=0.15, color ='green',label="Resnet34")
    plt.plot(steps18, smoothed_values18*100, color='purple', linewidth=2, alpha=1, label='Average')
    plt.plot(steps18, values18*100, alpha=0.15, color ='purple', label = "Resnet18")
    plt.xlabel(xaxis)
    plt.ylabel(yaxis +"(%)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'Data/ImagesOfResultGraphs/{title}.png')
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
    plt.savefig(f'Data/ImagesOfResultGraphs/{title}.png')
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

# def correctly_labeled(data):
#     count = 0
#     for i,ele in enumerate(data):
#         print(ele[1],ele[0])
#         if ele[1]==ele[0]:
#             count = count+1
#     print(count/i)

def correctly_labeled(data):
    data_to_azi_ele = defaultdict(list)
    with open('labels.csv', 'rt') as f:
            reader = csv.reader(f, delimiter=',') 
            for row in reader:
                data_to_azi_ele[row[2]]=(row[0],row[1])
    count_corr=0       
    count_azi_corr=0
    count_ele_corr=0

    for i,ele in enumerate(data):
        if ele[1]==ele[0]:
            count_corr += 1
        if int(data_to_azi_ele[str(ele[1][0])][0])==int(data_to_azi_ele[str(ele[0][0])][0]):
            count_azi_corr +=1
        if int(data_to_azi_ele[str(ele[1][0])][1])==int(data_to_azi_ele[str(ele[0][0])][1]):
            count_ele_corr +=1
    print("accuracy:",count_corr/i)
    print("azimuth accuracy:",count_azi_corr/i)
    print("elevation accuracy:",count_ele_corr/i)
#data = np.load("val_data5.npy").squeeze()

def tsne_plot_ipcl(data):
    data = torch.load(data)
    print(data.keys())
    label_azi_ele=[]
    values_to_match = [4,0,9]
    with open('labels.csv', 'rt') as f:
        reader = csv.reader(f, delimiter=',') 
        for row in reader:
            if int(row[2]) in values_to_match:
                label_azi_ele.append((row[0],row[1]))
  
    label_mapping = dict(zip(values_to_match,label_azi_ele))
    print(label_mapping)
    x_tensor = torch.tensor(values_to_match)
    mask = torch.isin(data['labels'], x_tensor)
    indexes = torch.nonzero(mask).squeeze()
    indexes_list = indexes.tolist()
    X = data['Embed'][indexes_list]
    Y = data['labels'][indexes_list]
    labels = [label_mapping[int(key)] for key in Y]
    print(len(indexes_list))
    print(Y)
    print(X.shape)
    tsne = TSNE(n_components=2,perplexity=45)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='viridis', s=100, alpha=0.7, edgecolor='k')
    plt.title('t-SNE Plot')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.legend()
  
    
def general_x_y_graphs_IPCL(data):
    result= torch.load(data)['perf_monitor']
    title = "Training Loss"
    train_loss = np.array(result['train_loss'])
    steps = np.array(result['epoch'])+1
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, alpha=1, color ='purple',label="Training loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'Data/ImagesOfResultGraphs/{title}.png')
    plt.show()
    
    
    
def IPCL(data):
    result= torch.load(data)
    resultdata = result
    print(type(result))
    print(resultdata['perf_monitor'])
    print(resultdata.keys())
    
def bar_accuracy():
    plt.bar(('ResNet 18','ResNet 34'),(43.07,43.85),color=('purple','green'),width=.8,edgecolor='black')
    plt.ylim(40,45)
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Test data')
    plt.savefig(f'Data/ImagesOfResultGraphs/Accuracy Test data.png')
    plt.show()
    

# data = np.load("test_data_Resnet18_30Epochs.npy")
# correctly_labeled(data)
# 0.430676913359301


# bar_accuracy()
# print(data.shape)
# print(data[:,0].shape)
# print(data[:,1].shape)
# classes_distribution_Dataset("Data/Train")
# plt.show()
# plt.show()
# print(data[:,1].shape)
# classes_distribution(data[:,1])
# classes_distribution(data[:,0])

# print(data.shape)
# confusion_matrix_graph_elevations(data)
# confusion_matrix_graph_azimuth(data)
#combine_confusion_matrix_elevations("Data/ImagesOfResultGraphs/ConfusionElevationResnet34")
#combine_confusion_matrix_azimuth("Data/ImagesOfResultGraphs/ConfusionAzimuthResnet34")


# loss_train_34 = "Data/ResultsCSV/Resnet34_30Epochs_Loss_train.csv"
# acc_train_34= "Data/ResultsCSV/Resnet34_30Epochs_Acc_train.csv"
# acc_val_34 = "Data/ResultsCSV/Resnet34_30Epochs_Acc_val.csv"
# loss_train_18 = "Data/ResultsCSV/Resnet18_30Epochs_Loss_train.csv"
# acc_train_18= "Data/ResultsCSV/Resnet18_30Epochs_Acc_train.csv"
# acc_val_18 = "Data/ResultsCSV/Resnet18_30Epochs_Acc_val.csv"
# general_x_y_graphs(loss_train_34,loss_train_18,"Training Loss","Loss","Step",20)
# general_x_y_graphs(acc_train_34,acc_train_18,"Training Accuracy","Acc","Step",20)
# general_x_y_graphs(acc_val_34,acc_val_18,"Accuracy validation","Acc","Epoch",4)

# loss_train ="Data/ResultsCSV/Resnet18_30Epochs_Loss_Train_PerEp.csv"
# loss_val = "Data/ResultsCSV/Resnet18_30Epochs_Loss_Val_PerEp.csv"
# train_val_accuracy_comparison(loss_train,loss_val,"Train vs Validation Loss  Resnet 18 30 Epochs","Loss","EPOCH")
# plt.show()
#data= get_azi_ele_value(data)
#spherePlotConfusionElevation(data)
# spherePlotConfusionAzimuth(data)
# plt.show()
#IPCL("results/ipcl0_ResNet18_SGD_CosineAnnealing_lars0_bs10_bm_16_ep20_out32_k512_n5_t0.7_lr0.0001.pth.tar")
general_x_y_graphs_IPCL("results/ipcl0_ResNet18_SGD_None_lars0_bs10_bm_16_ep20_out32_k512_n5_t0.7_lr0.001.pth.tar", "results/ipcl0_ResNet18_SGD_None_lars0_bs10_bm_16_ep20_out32_k512_n5_t0.7_lr0.001.pth.tar")
#tsne_plot_ipcl("results/ipcl0_ResNet18_SGD_CosineAnnealing_lars0_bs10_bm_16_ep20_out32_k512_n5_t0.7_lr0.0001.pth.tarcosine_embeddings_EP0")
# tsne_plot_ipcl("results/ipcl0_ResNet18_SGD_CosineAnnealing_lars0_bs10_bm_16_ep20_out32_k512_n5_t0.7_lr0.0001.pth.tarcosine_embeddings_EP2")
#tsne_plot_ipcl("results/ipcl0_ResNet18_SGD_CosineAnnealing_lars0_bs10_bm_16_ep20_out32_k512_n5_t0.7_lr0.0001.pth.tarcosine_embeddings_EP19")
plt.show()


