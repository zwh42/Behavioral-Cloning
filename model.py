import random
import os
import csv
import cv2
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dropout, ELU
from keras.layers.convolutional import Cropping2D, Convolution2D, MaxPooling2D
from keras.layers.core import Lambda, Dense, Activation, Flatten
import matplotlib.pyplot as plt
from collections import Counter

TRAIN_DATA_PATH = [r"C:\Localdata\data\behavioralclone\train_data_4_test"]

DO_VISUALIZE = False

'''
CSV file format
center,left,right,steering,throttle,brake,speed
center_2016_12_01_13_30_48_287.jpg, left_2016_12_01_13_30_48_287.jpg, right_2016_12_01_13_30_48_287.jpg, 0, 0, 0, 22.14829
center_2016_12_01_13_30_48_404.jpg, left_2016_12_01_13_30_48_404.jpg, right_2016_12_01_13_30_48_404.jpg, 0, 0, 0, 21.87963
'''

def data_preprocessing(csv_path_list):
    sample_line_list = []
    counter = Counter()
    
    for csv_path in csv_path_list:
        csv_name = os.path.join(csv_path, "driving_log.csv")
        with open(csv_name) as csvfile:
            next(csvfile, None) ## skip the header
            reader = csv.reader(csvfile)
            for line in reader:
                for i in range(3):
                    line[i] = os.path.join(csv_path, "IMG", line[i])
                sample_line_list.append(line)
                counter[float(line[3])] += 1

    print("sample list output example: ", sample_line_list[0])                
                
                

    
    keys = sorted(counter.keys())
    print("total sample count before cleaning: ", len(sample_line_list))
    
    most_common_angles = counter.most_common(10)
    print("most common steering angles:",most_common_angles)
    indexes = np.arange(len(keys)) 
    angle_count = [counter[k] for k in keys]
    
    if DO_VISUALIZE:    
        plt.bar(indexes, angle_count, width=2)
        plt.xticks(indexes + 0.5 * 0.5, keys, rotation=90)
        plt.show()
    
    
    clean_counter = Counter()
    final_sample_list = []
    
    #do sample cleaning 
    for sample in sample_line_list:
        if abs(float(sample[3]) - 0.00) < 0.001: #to many angle == 0, only chose part
            if random.random() <= most_common_angles[1][1]/most_common_angles[0][1] * 2:
                final_sample_list.append(sample)
                clean_counter[sample[3]] += 1
        else:
            final_sample_list.append(sample)
            clean_counter[sample[3]] += 1
    
    
    print("most common steering angles after cleaning:", clean_counter.most_common(10))
    clean_keys = sorted(clean_counter.keys())
    clean_indexes = np.arange(len(clean_keys)) 
    clean_angle_count = [clean_counter[k] for k in clean_keys]
    

    if DO_VISUALIZE:
        plt.bar(clean_indexes, clean_angle_count, width=2)
        plt.xticks(clean_indexes + 2 * 0.5, clean_keys, rotation=90)
        plt.show()
    
    print("total sample count after cleaning: ", len(final_sample_list))
    return final_sample_list
    

    
data_dict = data_preprocessing(TRAIN_DATA_PATH)



def generator(samples, batch_size=1000):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image_name = batch_sample[0]

                center_image = cv2.imread(center_image_name)
                center_angle = float(batch_sample[3])        

                #print(center_image)
                
                images.append(center_image)
                angles.append(center_angle)
                
                #flipping images And steering to avoid bias 
                flipped_center_image = np.fliplr(center_image)
                flipped_center_angle = -center_angle
                
                images.append(flipped_center_image)
                angles.append(flipped_center_angle)            

                
                
            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

            
print(next(generator(data_preprocessing(TRAIN_DATA_PATH))))





image_color_channel = 3
image_height = 160
image_width = 320


def model_setup():
    
    model = Sequential()

    #cropping layer
    crop_top = 70
    crop_bottom = 20
    crop_left = 10
    crop_right = 10
    model.add(Cropping2D(cropping=((crop_top,crop_bottom), (crop_left,crop_right)), input_shape=(160,320,3),  dim_ordering='tf'))
    
    #lambda layer: to normalize images to [-0.5, +0.5] 
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    
    
    #conv layer
    n_filters = 128
    model.add(Convolution2D(n_filters, 3, 3, border_mode = "valid"))
    
    #pooling 
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #drop out
    model.add(Dropout(0.5))
    
    #relu
    model.add(ELU())
    
    #conv layer
    model.add(Convolution2D(100, 3, 3, border_mode = "valid"))
    
    #pooling 
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #drop out
    model.add(Dropout(0.5))
    
    #relu
    model.add(ELU())
    
    
    #conv layer
    model.add(Convolution2D(64, 5, 5, border_mode = "same"))
    
    #pooling 
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #drop out
    model.add(Dropout(0.5))
    
    #relu
    model.add(ELU()) 
    
    '''
    #conv layer
    model.add(Convolution2D(32, 5, 5, border_mode = "valid"))
    
    #pooling 
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #drop out
    model.add(Dropout(0.5))
    
    #relu
    model.add(ELU())
    '''

    # flatten()
    model.add(Flatten())
    
    
    model.add(Dense(512))
    model.add(Dense(128))
    #drop out
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Dense(8))
    #output
    model.add(Dense(1))
    
    return model



def flow_setup():
    
    samples = data_preprocessing(TRAIN_DATA_PATH)
    print("total sample count:", len(samples))
    train_validation_samples, test_samples = train_test_split(samples, test_size = 0.2, random_state = 42)
    train_samples, validation_samples = train_test_split(train_validation_samples, test_size = 0.2, random_state = 42)
    print("train sample count: ", len(train_samples), "\nvalidation sample count: ", len(validation_samples), "\ntest sample count: ", len(test_samples))
    print("sample data example", train_samples[random.randint(0, len(train_samples))])
    
    train_generator = generator(train_samples, batch_size = 32)
    validation_generator = generator(validation_samples, batch_size = 32)
    test_generator = generator(test_samples, batch_size = 32)            
    #print("generator output example: \n", next(train_generator))
    
    model = model_setup()
    model.compile(loss = "mse", optimizer="adam")
    print(model.summary())
    
    history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
    score = model.evaluate_generator(test_generator, 1500, max_q_size=10, nb_worker=1, pickle_safe=False)
    #print(score)
    
    if DO_VISUALIZE:
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
    
    
    #model.save('model.h5')
    print("job finished. model saved")
    


if __name__ == "__main__":
    flow_setup()