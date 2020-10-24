# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 21:13:00 2020

@author: User
"""


# uploading an extracting zip files into directory in streamlit
import streamlit as st
from zipfile import ZipFile 
import os 
import numpy as np
import cv2
import glob
import itertools
from augment import AugImg



st.set_option('deprecation.showfileUploaderEncoding', False)

## io operations

def read_image(imagefile, dtype=np.float32):   
    img = cv2.imread(imagefile);
    return img

def save_image(image, tofile,index, format='.png', ):
   status= cv2.imwrite(tofile+str(index)+format ,image );
   return status

# reading and placing array in stack
def read_loop(files):
    img_list = []
    for x in files:
        img_list.append(read_image(x).flatten)
     # convert list to tuple 
    img_tuple=tuple(img_list)
    img_stack = np.stack(img_tuple, axis =0)
        
    return img_stack

# used to retireve specific extensions in folder
def getFilenames(exts):
    fnames=[glob.glob(ext) for ext in exts]
    fnames=list(itertools.chain.from_iterable(fnames))
    return fnames

# used to create the respective directory
def create_dir(process_dir):  
    for x in process_dir:        
         if not os.path.exists(x):
              os.makedirs(x)

   

     
def restop(files, count = 0):
        pathV = 'Vertical'
        pathCar = 'Cartoon'
        pathRC = 'Random_Crop'
        pathRMA = 'RotateMaxArea'
        pathH = 'Horizontal'
        pathJ = 'Color Jittering'
        pathS = "Sharp"
        pathMB = 'MotionBlur'
        pathCS = 'ChannelShift'
        process_dir=[pathV,pathRC,pathCar,pathRMA,pathH,pathJ,pathS, pathMB,pathCS]
        create_dir(process_dir)
        
        ai= AugImg();
        
        for x in files:
            if isinstance(read_image(x),np.ndarray):
                k=read_image(x)
                # perform vertical operation
                ve=ai.vertical_flip(k,1)
                # perform hortizontal operation
                ho=ai.horizontal_flip(k,0.8)
                # perfom cartooning of image
                car=ai.cartoon(k)
                # random crop
                rc=ai.random_crop(image=k,crop_size=100)
                # rotate-max-area
                rma=ai.rotate_max_area(image=k,angle=100)
                #jittering 
                jitter=ai.color_jittering(k)
                # sharpen
                sharp=ai.sharpen(k)
                #motion blur
                mb_v,mb_h=ai.motion_blur(k)


              


                save_image(mb_h, os.path.join(pathMB , 'motion_blur_h'), count)
                save_image(mb_v, os.path.join(pathMB , 'moion_blur_v'), count)
                save_image(sharp, os.path.join(pathS , 'sharp_'), count)
                save_image(ve, os.path.join(pathV , 'vertical_'), count)
                save_image(car, os.path.join(pathCar , 'cartoon_'), count)
                save_image(rc, os.path.join(pathRC , 'random_crop_'), count)
                save_image(rma, os.path.join(pathRMA , 'rma_'), count)
                save_image(ho, os.path.join(pathH , 'horizontal_'), count)
                save_image(jitter, os.path.join(pathJ , 'jitter_'), count)
                save_image(sharp, os.path.join(pathS , 'sharp_'), count)
                
                
                count+=1
                
if __name__ == '__main__':
    
    
    st.write('Get your data augmenteted here.')
    st.write('Size of data in zip file should be below 200MB')

    img_file_buffer = st.file_uploader("Upload your dataset in zip file here", type=["zip"])
    
    
    if img_file_buffer is not None:
        with ZipFile(img_file_buffer, 'r') as zip: 
            # printing all the contents of the zip file 
            st.write("File uploaded.")
            files = zip.namelist()
            # extracting all the files 
            zip.extractall() 
            # check if consist of all images and will filter the
            restop(files)
            st.write('Your data has been augmented, check your directory!')
    else:
        print('No files were passed in.')
        print('closing files')     
    


               