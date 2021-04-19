# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:46:41 2021

@author: natha
"""
#https://mpltest2.readthedocs.io/en/stable/gallery/event_handling/image_slices_viewer.html
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2


class IndexTracker:
    def __init__(self, ax, X,Y,A,B):
        self.ax = ax
        ax[0][0].set_title("cc mask")
        ax[0][1].set_title("fa mask")
        ax[1][0].set_title("fa mask")
        ax[1][1].set_title("rd mask")

        self.X = X
        self.Y = Y
        self.A = A
        self.B = B
        
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.imX = ax[0][0].imshow(self.X[:, :, self.ind], cmap='gray')
        self.imY = ax[0][1].imshow(self.Y[:, :, self.ind], cmap='gray')
        self.imA = ax[1][0].imshow(self.A[:, :, self.ind], cmap='gray')
        self.imB = ax[1][1].imshow(self.B[:, :, self.ind], cmap='gray')
        self.update()

    def on_scroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.imX.set_data(self.X[:, :, self.ind])
        self.imY.set_data(self.Y[:, :, self.ind])   
        self.imA.set_data(self.A[:, :, self.ind])
        self.imB.set_data(self.B[:, :, self.ind])
        self.ax[0][0].set_ylabel('slice %s' % self.ind)
        self.imX.axes.figure.canvas.draw()

file_path = "Nifti_Brain/E1_FA.nii"
fa_img=nib.load(file_path)
data=fa_img.get_fdata()

file_path = "Nifti_Brain/E1_AD.nii"
ad_img=nib.load(file_path)
ad_data = ad_img.get_fdata()

file_path = "Nifti_Brain/E1_MD.nii"
md_img=nib.load(file_path)
md_data = md_img.get_fdata()

file_path = "Nifti_Brain/E1_RD.nii"
rd_img=nib.load(file_path)
rd_data = rd_img.get_fdata()

file_path = "Nifti_Brain/E1_color_fa.nii"
fac_img=nib.load(file_path)
fac_data = fac_img.get_fdata()

file_path = "Nifti_Brain/BrainMask_(1.42, 2.32, 3.76).nii"
bm_img=nib.load(file_path)
bm_data = bm_img.get_fdata()
print("bm_data")
print(bm_data.shape)

morpho = True
threshold = 0.61

bm_mask = np.zeros(bm_data.shape)

for i in range(data.shape[2]):
    bm_mask[:,:,i] = bm_data[:,:,i]

fa_mask = np.zeros(data.shape)

#Threshold mask for fa_data
print("data max: " + str(np.max(data)))
fa_threshold = np.percentile(data.reshape(-1),99)

print("fa_threshold: " + str(fa_threshold))
for i in range(data.shape[2]):
    img = data[:,:,i]
    
    thresh = np.where(img > fa_threshold, 1, img/3) #0.675 gives good results

    if morpho:
        #Apply morphological operators
        kernel = np.ones((1,2),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((2,1),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    fa_mask[:,:,i] = thresh
    

rd_mask = np.zeros(data.shape)

#Threshold mask for rd_data
for i in range(data.shape[2]):
    img = rd_data[:,:,i]
    #Apply threshold to FA intensity
    min_thresh = np.where(img > 0, 1, img/2)
    max_thresh = np.where(img < 0.00047, 1,img/3)
    
    bool_mask = np.equal(min_thresh, max_thresh)
    thresh = np.where(bool_mask == True, 1, img)

    if morpho:
        #Apply morphological operators
        kernel = np.ones((1,2),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((2,1),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    rd_mask[:,:,i] = thresh



fac_mask = np.zeros(data.shape)

for i in range(data.shape[2]):
    img = fac_data[:,:,i,0]

    if i == 0:
        print("max: " + str(np.max(img)))
        print(sum(sum(img)))
    
    #Apply threshold to FA intensity
    thresh = np.where(img > 0.4, 1, img/2 + 0.0001) #0.675 gives good results
    if i == 0:
        print(np.max(thresh))
        
    if morpho:
        #Apply morphological operators
        kernel = np.ones((1,2),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((2,1),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    fac_mask[:,:,i] = thresh

#fac AND rd mask
cc_mask = np.equal(fac_mask, rd_mask)
clipped_img = nib.Nifti1Image(cc_mask, fac_img.affine, fac_img.header)
nib.save(clipped_img, 'facANDrd_mask.nii')


fig, ax = plt.subplots(2, 2)
#print(ax.shape)

tracker = IndexTracker(ax,cc_mask, fa_mask,fac_mask, rd_mask)

fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()  