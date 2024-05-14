from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
import csv
from sklearn import preprocessing

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

    ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cm.hot(myheatmap))

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
    norm=preprocessing.normalize(np.array(azi_ele))
    
    plt.imshow(norm)
    print(norm)
    return norm
data = np.load("val_data2.npy")
print(data.shape)
data= get_azi_ele_value(data)
spherePlot(data)