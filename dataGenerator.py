
import csv
import os
import subprocess
import shutil
import re
from pydub import AudioSegment
import numpy as np
import csv
import os
import subprocess
import shutil
import re
import numpy as np
import torch
import torchaudio
from df.enhance import enhance, init_df, load_audio, save_audio 
import noisereduce as nr

import librosa

def renameData(csv_file, dataset):
    majcat_dic = {"animals":["dog","rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"],
            "natural": ["rain","sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm"],
             "human": ["crying_baby","sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping"],
             "domestic": ["door_wood_knock","mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking"],
             "urban": ["helicopter","chainsaw", "siren", "car_horn", "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw"]
    }
    cat_count = {}
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            original_name = row['filename']
            cat= row['category']
            if cat in cat_count:
                cat_count.update({cat:cat_count[cat]+1})
            else:
                cat_count.update({cat:1})
        
            for category, sounds in majcat_dic.items():
                if cat in sounds:
                    majcat= category
            new_name = f"{majcat}_{cat}_{cat_count[cat]}.wav"
            original_file_path = os.path.join(dataset, original_name)
            new_file_path = os.path.join(dataset, new_name)
            if os.path.exists(original_file_path):
                os.rename(original_file_path, new_file_path)
                print(f"Renamed {original_name} to {new_name}")
            else:
                print(f"File {original_name} not found!")

def shortening(dataset):
    files = os.listdir(dataset)
    for file in files:
        input_file = os.path.join(dataset, file)
        output_file = os.path.join("ESC-50-master/audio1s", file)
        # Execute ffmpeg command to trim the file
        command = f"ffmpeg -y -ss 0 -t 1 -i {input_file} {output_file}"
        subprocess.run(command, shell=True)
        print(f"Trimmed {file} to 1 second.")

def shortening2(dataset):
    files = os.listdir(dataset)
    for file in files:
        input_file = os.path.join(dataset, file)
        output_file = os.path.join("ESC-50-master/audioNoS2", file)
        sound, sample_rate = torchaudio.load(input_file)
        start_trim = detect_leading_silence(sound)
        duration = len(sound[0])    
        command = f"ffmpeg -y -ss {round(start_trim/sample_rate,2)} -t {round(duration/sample_rate,2)} -i {input_file} {output_file}"
        subprocess.run(command, shell=True)
        print(f"Removed slience{file}")

def detect_leading_silence(sound, silence_threshold=0.5, chunk_size=75):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms
    assert chunk_size > 0 # to avoid infinite loop
    avg = torch.sqrt(torch.mean(sound[:,:]**2))
    avgthresh= avg*silence_threshold
    print(f"avg:{avg}")
    print(avgthresh)
    while torch.sqrt(torch.mean(sound[:,trim_ms:trim_ms+chunk_size]**2)) < avgthresh and trim_ms < len(sound[0]):
        trim_ms += chunk_size
    return max(trim_ms-3000,0)



def training(dataset):
    for file in os.listdir(dataset):
        fileNum = re.findall(r'\d+',file)
        if int(fileNum[-1]) <11:
            shutil.copyfile(os.path.join(dataset, file),os.path.join("ESC-50-master/training", file))

            
def test(dataset):
    for file in os.listdir(dataset):
        fileNum = re.findall(r'\d+',file)
        if 11<=int(fileNum[-1])<13 :
            shutil.copyfile(os.path.join(dataset, file),os.path.join("ESC-50-master/test", file))

def renameRemove1(dataset):
    for file in os.listdir(dataset):
        original_file_path = os.path.join(dataset, file)
        new_file_path = os.path.join(dataset, file.replace("1","",1) )
        os.rename(original_file_path,new_file_path)
        
def dataset_to_ImageFolder(path):
    destination = "Data/DataIPCL/ImageFolderDatasetTrainNew"
    i = 0
    if not os.path.exists(destination):
        os.makedirs(destination)
    for filename in os.listdir(path):
        pattern = r"Az_(?P<azimuth>-?\d+)_El_(?P<elevation>-?\d+)"
        match = re.search(pattern,filename)
        azimuth = int(match.group('azimuth'))
        elevation = int(match.group('elevation'))
        if match:
            classlabel = f"{i}_position_{azimuth}_{elevation}"
            classdir = os.path.join(destination,f"class_{classlabel}")
            if not os.path.exists(classdir):
                os.makedirs(classdir)
                i=i+1
                
            source = os.path.join(path, filename)
            destination_new = os.path.join(classdir, filename)
            shutil.move(source, destination_new)

def add_label_to_class_dir(path):
    for i,dir in enumerate(os.listdir(path)):
        os.rename(os.path.join(path,dir),os.path.join(path,f"{i}_{dir}"))
    
#renameRemove1("ESC-50-master/test")
#test("ESC-50-master/audio1s")
#training("ESC-50-master/audio1s")
#shortening("ESC-50-master/audioNoS2")
#shortening2("ESC-50-master/audio")
#renameData("ESC-50-master/meta/esc50.csv", "ESC-50-master/audio")

        # path, target = self.imgs[index]
        # img = np.load(path)['arr_0']
        # files = os.listdir(os.path.dirname(path))
        # pattern = r"Az_(?P<azimuth>-?\d+)_El_(?P<elevation>-?\d+)"
        # labelData = re.search(pattern, path)
        # azimuth = int(labelData.group('azimuth'))
        # elevation = int(labelData.group('elevation'))
        # same_label = [p for p in files if re.search(r"Az_"+azimuth+"_El_"+elevation)]
        # img = [np.load(random.choice(same_label))['arr_0'] for i in range(self.n_samples)]

# test = np.load('ESC-50-master/Cochleagrams_PaulZimmer/Room_01_RT60_1.0_gx_1.5_gy_3.5_gz_1.5_Az_000_El_020_urban_train_5/arr_0.npy')
# print(test.shape)
add_label_to_class_dir("Data/DataIPCL/ImageFolderDatasetVal")



                