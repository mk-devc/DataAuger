# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 08:07:39 2020

@author: User
"""
# data augmnentation using opencv 


import math
import cv2
import numpy as np


class AugImg:   
    def __init__(self):
        print('Begining process image.')

       
    def horizontal_flip(self,image, rate=0.5):      
       if np.random.rand() < rate:
            image = image[:, ::-1, :]
       return image
    
    def vertical_flip(self,image, rate=0.5):
        if np.random.rand() < rate:
            image = image[::-1, :, :]
        return image
    def motion_blur(self,image):
        # Specify the kernel size. 
        # The greater the size, the more the motion. 
        kernel_size = 30
          
        # Create the vertical kernel. 
        kernel_v = np.zeros((kernel_size, kernel_size)) 
          
        # Create a copy of the same for creating the horizontal kernel. 
        kernel_h = np.copy(kernel_v) 
          
        # Fill the middle row with ones. 
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
          
        # Normalize. 
        kernel_v /= kernel_size 
        kernel_h /= kernel_size 
          
        # Apply the vertical kernel. 
        vertical_mb = cv2.filter2D(image, -1, kernel_v) 
          
        # Apply the horizontal kernel. 
        horizonal_mb = cv2.filter2D(image, -1, kernel_h) 
        
        return vertical_mb , horizonal_mb
    
    def sharpen(self,img):
        fil = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # Applying cv2.filter2D function on our Logo image
        sharpen_img=cv2.filter2D(img,-1,fil)
        return sharpen_img
    

    
    def rotate_bound(self,image, angle):
        # CREDIT: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(image, M, (nW, nH))
    
    
    def rotatedRectWithMaxArea(self,w, h, angle):
      """
      Given a rectangle of size wxh that has been rotated by 'angle' (in
      radians), computes the width and height of the largest possible
      axis-aligned rectangle (maximal area) within the rotated rectangle.
      """
      if w <= 0 or h <= 0:
        return 0,0
    
      width_is_longer = w >= h
      side_long, side_short = (w,h) if width_is_longer else (h,w)
    
      # since the solutions for angle, -angle and 180-angle are all the same,
      # if suffices to look at the first quadrant and the absolute values of sin,cos:
      sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
      if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
      else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    
      return wr,hr
     
    
    def rotate_max_area(self,image, angle):
        """ image: cv2 image matrix object
            angle: in degree
        """
        wr, hr = self.rotatedRectWithMaxArea(image.shape[1], image.shape[0],
                                        math.radians(angle))
        rotated = self.rotate_bound(image, angle)
        h, w, _ = rotated.shape
        y1 = h//2 - int(hr/2)
        y2 = y1 + int(hr)
        x1 = w//2 - int(wr/2)
        x2 = x1 + int(wr)
        return rotated[y1:y2, x1:x2]
    
    
    def random_crop(self,image, crop_size):
        crop_size = self.check_size(crop_size)
        h, w, _ = image.shape
        top = np.random.randint(0, h - crop_size[0])
        left = np.random.randint(0, w - crop_size[1])
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        image = image[top:bottom, left:right, :]
        return image
    
    
    
    def check_size(self,size):
        if type(size) == int:
            size = (size, size)
        if type(size) != tuple:
            raise TypeError('size is int or tuple')
        return size
    
    
    def scale_aug(self,image, scale_range, crop_size):
        scale_size = np.random.randint(*scale_range)
        image = np.array(cv2.resize(image , (scale_size,scale_size)))
        #image = imresize(image, (scale_size, scale_size))
        image = self.random_crop(image, crop_size)
        return image
    
    def cartoon(self,image):
        
        img_rgb = image
        num_down = 0 # number of downsampling steps
        num_bilateral = 7 # number of bilateral filtering steps
    
        
        # downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(num_down):
           img_color = cv2.pyrDown(img_color)
        
        # repeatedly apply small bilateral filter instead of
        # applying one large filter
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        
        # upsample image to original size
        for _ in range(num_down):
           img_color = cv2.pyrUp(img_color)
        
        #STEP 2 & 3
        #Use median filter to reduce noise
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        
        #STEP 4
        #Use adaptive thresholding to create an edge mask
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
           cv2.ADAPTIVE_THRESH_MEAN_C,
           cv2.THRESH_BINARY,
           blockSize=9,
           C=2)
        
        
        
        # Step 5
        # Combine color image with edge mask & display picture
        # convert back to color, bit-AND with color image
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        print(img_edge.shape)
        print(img_color.shape)
        img_cartoon = cv2.bitwise_and(img_color, img_edge)
        
        return img_cartoon
    
    
    def color_jittering(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,c = img.shape 
        
        noise = np.random.randint(0,50,(h, w)) 
        jitter_add = np.zeros_like(img)
        jitter_add[:,:,1] = noise  
        
        # convert h to int for indexing
        ih=int(h/2)
        
        noise_added = cv2.add(img, jitter_add)
        combined = np.vstack((img[:ih,:,:], noise_added[ih:,:,:]))
        
        return combined
    
    

