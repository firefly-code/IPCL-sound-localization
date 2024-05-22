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


def spherePlot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, np.pi, 72)

    # create the sphere surface
    x=1.5 * np.outer(np.cos(u), np.sin(v))
    y=1.5 * np.outer(np.sin(u), np.sin(v))
    z=1.5 * np.outer(np.ones(np.size(u)), np.cos(v))

    # simulate heat pattern (striped)
    print(np.abs(np.sin(y)).shape)
    #grid[labels]= probility
    myheatmap = data
    print(myheatmap.shape)

    ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cm.BuGn(myheatmap))

    plt.show()
    
def get_azi_ele_value(data):
    azi_ele= np.zeros((72,72))
    for label,pred in data:
        with open('labels.csv', 'rt') as f:
                reader = csv.reader(f, delimiter=',') 
                for row in reader:
                    if(int(row[2])==label):
                        # print("lab2:",row[0],row[1])
                        azi = (int(int(row[0])/5))-1
                        ele = (int((180+int(row[1]))/5))-1
                        # print("lab:",row[2])
                        # print("azi:",azi)
                        # print("ele:",ele)
                        if(azi_ele[azi,ele] != 0):
                            azi_ele[azi,ele]= (azi_ele[azi,ele]+(label-pred))/2
                        else:
                            azi_ele[azi,ele]= (label-pred)
    norm = gaussian_filter(np.array(azi_ele), sigma=2)
    norm=preprocessing.normalize(norm)
    plt.imshow(norm)
    return norm

def confusion_matrix_graph(data):
    cm = confusion_matrix(data[:,0],data[:,1],normalize='all')
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Greens', square=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.xticks(np.arange(190), rotation=90)
    plt.yticks(np.arange(190), rotation=0)
    plt.show()

def general_x_y_graphs(data_path,title,yaxis,xaxis,smoothing):
    csv_file_path = data_path
    data = pd.read_csv(csv_file_path)

    # Extract the relevant columns
    steps = data['Step']
    values = data['Value']
    window_size = smoothing 
    smoothed_values = values.rolling(window=window_size).mean()
    # Plot the data
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

    # Extract the relevant columns
    steps = data_train['Step']
    values_train = data_train['Value']
    values_val =data_val['Value']
    
    window_size = 4
    smoothed_values_train = values_train.rolling(window=window_size).mean()
    smoothed_values_val = values_val.rolling(window=window_size).mean()
    # Plot the data
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
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=190, edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Numbers')
    plt.grid(True)
    


data = np.load("val_data2.npy").squeeze()
# print(data[:,1].shape)
# classes_distribution(data[:,1])
# classes_distribution(data[:,0])
# plt.show()
# print(data.shape)
# confusion_matrix_graph(data)

# acc_train = "Data/ResultsCSV/run-May09_00-36-08_ValkBig-tag-Acc_train.csv"
# acc_val = "Data/ResultsCSV/run-May09_00-36-08_ValkBig-tag-Acc_Val.csv"

# general_x_y_graphs(acc_train,"Accuracy training","Acc","Step",20)
# general_x_y_graphs(acc_val,"Accuracy validation","Acc","Epoch",4)
loss_train ="Data/ResultsCSV/run-resnet_ssl20240509_003608_Training vs. Validation Loss_Training-tag-Training vs. Validation Loss.csv"
loss_val = "Data/ResultsCSV/run-resnet_ssl20240509_003608_Training vs. Validation Loss_Validation-tag-Training vs. Validation Loss.csv"
train_val_accuracy_comparison(loss_train,loss_val,"Train vs Validation Loss","Loss","EPOCH")


# data= get_azi_ele_value(data)
# spherePlot(data)


