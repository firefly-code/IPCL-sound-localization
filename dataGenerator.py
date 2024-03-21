
import csv
import os
import subprocess
import shutil
import re
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
    
renameRemove1("ESC-50-master/test")

#test("ESC-50-master/audio1s")
#training("ESC-50-master/audio1s")
#shortening("ESC-50-master/audio")
#renameData("ESC-50-master/meta/esc50.csv", "ESC-50-master/audio")
        


                