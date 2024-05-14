import csv

with open('labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    azimuth = [270,280,290,300,310,320,330,340,350,0,10,20,30,40,50,60,70,80,90]
    elvation = [-45,-30,-20,-10,0,10,20,30,45,60]
    label =0 
    for i in range(19):
        for j in range(10):
            writer.writerow([azimuth[i],elvation[j],label])
            label +=1 
    

