import os
import sys
import glob
import random
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2
import albumentations as album

class DataLoader_test:
    def __init__(self, directory, images_dir='',labels_file='',task_augmentation_dir=''):
        self.directory = directory
        self.task_aug_dir = task_augmentation_dir
        self.images_dir = images_dir
        self.labels_file= labels_file

    def Load_all_batches(self, data, data_shape=(-1,1)):
        loaded_set_features = []
        loaded_set_labels= []
        for batch in data:
            X = []
            Y = []
            for instance in batch:
                link = instance[0]
                image = Image.open(link)
                data = asarray(image)
                if data.shape[2] == 4:
                    data = data[:,:,:-1]
                if data_shape != 'original':
                    data = data.reshape(data_shape)
                X.append(data)
                image_number = instance[1]
                angle = instance[2]
                speed = instance[3]
                if float(speed) < 0.5: 
                    speed = np.random.uniform(0,0.15)
                elif float(speed) >= 0.5:
                    speed = np.random.uniform(0.85,1)
                Labels = np.array([angle,speed], dtype=np.float32)
                Y.append(Labels)
            loaded_set_features.append(X)
            loaded_set_labels.append(Y)
        loaded_set_features = np.array(loaded_set_features,dtype=np.uint8)
        loaded_set_labels= np.array(loaded_set_labels,dtype=np.float32)
        return loaded_set_features,loaded_set_labels

    def Load_batch(self, batch, data_shape=(-1,1), augmentation=False, 
                   augmentation_pass = 1,augmentation_protocol=[], 
                   augmentation_size=256):
        X = []
        Y = []
        for instance in batch:
            link = instance[0]
            image = Image.open(link)
            data = asarray(image)
            if data.shape[2] == 4:
                data = data[:,:,:-1]
            if data_shape != 'original':
                data = data.reshape(data_shape)
            X.append(data)
            image_number = instance[1]
            angle = instance[2]
            speed = instance[3]
            if float(speed) < 0.5: 
                speed = np.random.uniform(0,0.15)
            elif float(speed) >= 0.5:
                speed = np.random.uniform(0.85,1)
            Labels = np.array([angle,speed], dtype=np.float32)
            Y.append(Labels)
        if augmentation:
            X,Y = self._augment_batch(X,Y, n=augmentation_pass,
                                      protocol=augmentation_protocol
                                      ,target_aug_batch_size = augmentation_size)
        loaded_set_features = np.array(X,dtype=np.uint8)
        loaded_set_labels= np.array(Y,dtype=np.float32)
        return loaded_set_features,loaded_set_labels

    def Load_batch_quiz(self, batch, data_shape=(-1,1)):
        X = []
        Image_order = []
        for instance in batch:
            link = instance[0]
            image = Image.open(link)
            data = asarray(image)
            if data.shape[2] == 4:
                data = data[:,:,:-1]
            if data_shape != 'original':
                data = data.reshape(data_shape)
            image_number = instance[1]
            X.append(data)
            Image_order.append(image_number)
        X = np.array(X)
        Image_order = np.array(Image_order)
        return X,Image_order

    def LoadCollectedData_info(self,split,batch_size):
        self._get_collected_imgsinfo_train()
        if split:
            self._train_test_split(split=split)
        if batch_size:
            self._batch_imgsinfo_train(batch_size=batch_size)
        return self.train_images_information, np.array([self.test_images_information])

    def LoadModelData_info(self, split, batch_size):
        self._get_imgsinfo_train()
        if split:
            self._train_test_split(split=split)
        if batch_size:
            self._batch_imgsinfo_train(batch_size=batch_size)
        return self.train_images_information, np.array([self.test_images_information])

    def LoadQuizData_info(self):
        self._get_imgsinfo_test()
        return self.quiz_images_information

    def _train_test_split(self, split):
        n = len(self.train_images_information)
        split_index = int(np.ceil(n * split))
        train_split = self.train_images_information[0:split_index]
        test_split = self.train_images_information[split_index+1:n]
        self.train_images_information = train_split
        self.test_images_information = test_split

    def _batch_imgsinfo_train(self, batch_size):
        n = len(self.train_images_information)
        if n % batch_size ==0:
            batched_imgsinfo_train = np.reshape(self.train_images_information,
                    (int(n/batch_size),-1,4)
                    )
        else: 
            while n % batch_size != 0:
                n -= 1
            batched_imgsinfo_train = np.reshape(
                    self.train_images_information[0:n],
                    (int(n/batch_size),-1,4)
                    )
        self.train_images_information = batched_imgsinfo_train

    def _get_collected_imgsinfo_train(self):
        #Get the absolute paths of the data
        absolute_paths = glob.glob(self.directory+"/*.png") 
        #getting the labels
        self.train_images_information= []
        for path in absolute_paths:
            parts = path.split('/')[-1].split('.')[0].split('_')
            image_number= parts[0]
            angle = parts[1]
            angle = round((int(angle)-50)/(80),3)
            speed = parts[2]
            speed = round(int(speed)/35, 3)
            if speed < 0.5: 
                speed = np.random.uniform(0,0.15)
            elif speed >= 0.5:
                speed = np.random.uniform(0.85,1)
            information =[ 
                    path,
                    image_number,
                    angle,
                    speed
                    ]
            self.train_images_information.append(information)

    def _get_imgsinfo_train(self):
        #Get the absolute paths of the data
        absolute_paths = glob.glob(self.directory+self.images_dir+"/*.png") 
        #getting the labels
        labels = self._get_labels()
        df = pd.DataFrame(labels, columns = ['index','angle','speed'])
        df = df.set_index('index')
        self.train_images_information= []
        for path in absolute_paths:
            parts= path.split('/')
            image_name = parts[-1]
            image_number = int(image_name.split('.')[0])
            if image_number in df.index:
                image_label = df.loc[image_number]
            else:
                continue
            angle = image_label[0]
            speed = image_label[1]

            information =[ 
                    path,
                    image_number,
                    angle,
                    speed
                    ]
            self.train_images_information.append(information)

    def _get_imgsinfo_test(self):
        #Get the absolute paths of the data
        absolute_paths = glob.glob(self.directory+self.images_dir+"/*.png") 
        #getting the labels
        self.quiz_images_information= []
        for path in absolute_paths:
            parts= path.split('/')
            image_name = parts[-1]
            image_number = int(image_name.split('.')[0])
            information =[ 
                    path,
                    image_number,
                    ]
            self.quiz_images_information.append(information)

    def _get_labels(self):
        labels = genfromtxt(self.directory+self.labels_file , delimiter=',' , skip_header=1, dtype=np.float32) 
        return labels
        
        
    #augmentation_protocol options
    ## Normal, TurnBias, StopGoLights, SyntheticStopGoLights,
    ## StopForObjec, MirrorSteeringFlip
    def _augment_batch(self, X,Y,n,protocol,target_aug_batch_size):
        Xs = []
        Ys = []
        sx,sy = 90,140
        pro_copy = protocol
        for _ in range(n):
            for x,y in zip(X,Y):
                if len(Xs) < target_aug_batch_size:
                  x, y= self._unit_augment_album_invert(x,y)
                  Xs.append(x)
                  Ys.append(y)
                  protocol = [random.choice(tuple(pro_copy))]
                  if 'Normal' in protocol:
                    x_album, y_album = self._unit_augment_album(x,y)
                    Xs.append(x_album)
                    Ys.append(y_album)
                  if 'TurnBias' in protocol:
                    x_turn, y_turn = self._unit_augment_turn(x,y,sx,sy)
                    x_album, y_album = self._unit_augment_album(x_turn,y_turn)
                    Xs.append(x_album)
                    Ys.append(y_album)
                  if 'StopGoLights' in protocol:
                    x_light, y_light = self._unit_augment_light(x,y,sx,sy)
                    x_album, y_album = self._unit_augment_album(x_light,y_light)
                    Xs.append(x_album)
                    Ys.append(y_album)
                  if 'SyntheticStopGoLights' in protocol:
                    x_slight, y_slight = self._unit_augment_slight(x,y)
                    x_album, y_album = self._unit_augment_album(x_slight,y_slight)
                    Xs.append(x_album)
                    Ys.append(y_album)
                  if 'StopForObjec' in protocol:
                    x_ostop, y_ostop = self._unit_augment_object_stop(x,y,sx,sy)
                    x_album, y_album = self._unit_augment_album(x_ostop,y_ostop)
                    Xs.append(x_album)
                    Ys.append(y_album)
                  if 'MirrorSteeringFlip' in protocol:
                    x_mirror,y_mirror = self._unit_augment_mirror(x,y)
                    x_album, y_album = self._unit_augment_album(x_mirror,y_mirror)
                    Xs.append(x_album)
                    Ys.append(y_album)
                else:
                  pass
        if len(Xs) > target_aug_batch_size:
          n = len(Xs) - target_aug_batch_size
          def delete_rand_items(Xs,Ys,n):
            to_delete = set(random.sample(range(len(Xs)),n))
            return [x for i,x in enumerate(Xs) if not i in to_delete],[y  for i,y in enumerate(Ys) if not i in to_delete] 
          Xs,Ys = delete_rand_items(Xs,Ys,n)
        return Xs,Ys
    
    def _unit_augment_album(self, x,y):
        transform = album.Compose([
            album.OneOf([
                album.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 7)),
                album.RandomBrightnessContrast(always_apply=False, p=1.0, 
                                                brightness_limit=(-0.2, 0.2), 
                                                contrast_limit=(-0.2, 0.2), brightness_by_max=True),
                album.Rotate(always_apply=False, p=1.0, limit=(-31, 31), 
                                interpolation=0, border_mode=1, 
                                value=(0, 0, 0), mask_value=None),
                album.RandomResizedCrop(always_apply=False, p=1.0, 
                                        height=240, width=320, scale=(0.5399999618530273, 1.0), 
                                        ratio=(0.7299999594688416, 2.259999990463257), interpolation=1),

            ])
        ])
        x_basic, y_basic = transform(image=x)['image'], y
        return x_basic, y_basic

    def _unit_augment_album_invert(self, x,y):
        transform = album.Compose([
            album.InvertImg(always_apply=False, p=1.0)
        ])
        x_basic, y_basic = transform(image=x)['image'], y
        return x_basic, y_basic
    
    def _unit_augment_turn(self,x,y,sx,sy):
        if y[0] >0.4 and y[0] <0.6:
            objects = ['steer_left.png','steer_right.png']
            rand = np.random.randint(0,1)
            input = Image.open(os.path.join(self.task_aug_dir,objects[rand]))
            input = np.asarray(input.resize((sx,sy)))[:,:,:]
            #x = cv2.cvtColor(x, cv2.COLOR_RGB2RGBA)
            region=(0,(320-sx-5),0,(240-sy-5))
            x_new = self._placeimage_A_in_B(input,x,region=region)
            y = y.copy()
            if rand ==0:
                y[0] = np.random.uniform(0,0.15)
            elif rand ==1:
                y[0] = np.random.uniform(0.85,1)
            return x_new,y
        else:
            return x,y
        
    
    def _unit_augment_light(self,x,y,sx,sy):
      objects = ['green_light.png','red_light.png']
      rand = np.random.randint(0,1)
      input = Image.open(os.path.join(self.task_aug_dir,objects[rand]))
      input = np.asarray(input.resize((sx,sy)))[:,:,:]
      #x= cv2.cvtColor(x, cv2.COLOR_RGB2RGBA)
      region=(0,(320-sx-5),0,(240-sy-5))
      x_new = self._placeimage_A_in_B(input,x,region=region)
      y = y.copy()
      if rand ==0:
        y[1] = 1
      elif rand ==1:
        y[1] = 0
      return x_new,y
    
    def _unit_augment_slight(self, x,y):
        #takes a single image and mirrors it while flipping the steering angle
        def rand_light_gen():
            #creates random light, returns light and status
            light_size = np.random.randint(10,20)
            _light = np.zeros((light_size,light_size,3))
            red_light = _light[:] + (255,0,0)
            green_light = _light[:] + (0,255,0)
            rand = np.random.uniform(0,1)
            if rand >= 0.5:
                return green_light,'go'
            else:
                return red_light,'stop'
        def add_light_rand(image):
            #adds light to image, returns image
            light, status = rand_light_gen()
            size = light.shape[0]
            light_field_x = int(np.round(image.shape[0]*0.6))
            light_field_y = int(image.shape[1])
            x_start = np.random.randint(0+(size+1),light_field_x-(size-1))
            x_end = x_start + size
            y_start = np.random.randint(0+(size+1),light_field_y-(size-1))
            y_end = y_start + size
            image = image.copy()
            image[x_start:x_end,y_start:y_end,:] = light
            return image, status
        x_new, status = add_light_rand(x)
        y = y.copy()
        if status == 'stop':
            y[1] = 0
        else:
            y[1] = 1
        return x_new,y
    
    def _unit_augment_object_stop(self, x,y, sx,sy):
        if y[1] > 0.8:
            #takes a single image and mirrors it while flipping the steering angle
            objects = ['person_stop.png','tree_stop.png','person_stop2.png','box_stop.png']
            rand = np.random.randint(0,4)
            input = Image.open(os.path.join(self.task_aug_dir,objects[rand]))
            input = np.asarray(input.resize((sx,sy)))[:,:,:]
            #x= cv2.cvtColor(x, cv2.COLOR_RGB2RGBA)
            region=(140,160,90,(240-sy))
            x_new = self._placeimage_A_in_B(input,x,region=region)
            y = y.copy()
            y[1] = 0
            return x_new,y
        else:
            return x,y
    
    def _unit_augment_mirror(self, x,y):
        #takes a single image and mirrors it while flipping the steering angle
        transform = album.Compose([
            album.HorizontalFlip(p=1),
        ])
        new = transform(image=x)['image']
        y = y.copy()
        y[0] = (y[0]*-1) +1 
        return new,y
        
    def _placeimage_A_in_B(self,A,B, region):
            #adds light to image, returns image
            input = Image.fromarray(np.uint8(A)) #B 
            image = Image.fromarray(np.uint8(B)) #B 
            # randomised placement
            size = A.shape
            x_start = np.random.randint(region[0],region[1])
            y_start = np.random.randint(region[2],region[3])
            

            image.paste(input, (x_start, y_start), input)
            output = np.asarray(image)
            return output




