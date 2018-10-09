# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:48:57 2018

"""

#importing all necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Sequential
from scipy.misc import imread
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Dense
import pandas as pd
import cv2
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from scipy.misc import imresize
from imutils import face_utils
import cv2
import os
from keras_vggface.vggface import VGGFace

import math as mt
import sys
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics

from scipy import misc
import tensorflow as tf
import align.detect_face
import time
import dlib



def Intelligent_Vision():
    start_time = time.time()
    # setup facenet parameters
    gpu_memory_fraction = 1.0
    minsize = 20 # minimum size of face
    #Given an image, we initially resize it to different scales to build an image pyramid, 
    #which is the input of the following three-stage cascaded framework: 
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    #factor = 0.709 # scale factor
    factor = 0.45# scale factor
    column = ['croped_images','timestamp']
    database= pd.DataFrame(columns=column)  #creating dataframe for each face and their respective timestamp
    index=0
    
    
    # fetch the video
    path,dirs,files = os.walk("F:\\Aegis\\Capstone Project\\Capstone Project\\Final Project\\Final_Running_Code\\uploads\\new_video").__next__()
    cap = cv2.VideoCapture(path+'\\'+files[0])
    
    #   Start code from facenet/src/compare.py
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
            log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(
                sess, None)
    #end code from facenet/src/compare.py
    f = 39
    index= 0
    while 1:
        ret, img = cap.read()
        if(ret==False):
            break
        #run detect_face from the facenet library
        f=f+1
        if (f==40):
            bounding_boxes, _ = align.detect_face.detect_face(
            img, minsize, pnet,
            rnet, onet, threshold, factor)
            #   for each box
            for (x1, y1, x2, y2, acc) in bounding_boxes:
                w = x2-x1
                h = y2-y1
                
                crop_img = img[int(y1): int(y1 + h), int(x1): int(x1 + w)]
                database.loc[index]=[crop_img,cap.get(cv2.CAP_PROP_POS_MSEC)]
                index=index+1
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            f=0
                  
    cap.release()
    cv2.destroyAllWindows()
    
    ##removing the entry for the images  that has zero length
    drop_list = list()
    for i in range(len(database)):
        if (len(database.iloc[i,0])==0):
            drop_list.append(i)
            
    database.drop(drop_list, inplace = True)
    database = database.reset_index(drop=True)
    index = len(database)
    
    
    # reading image to be matched and saving it along with timestamp to the dataframe
    path,dirs,files = os.walk("F:\\Aegis\\Capstone Project\\Capstone Project\\Final Project\\Final_Running_Code\\uploads\\new_image").__next__()
    
    match_image= cv2.imread(path+'\\'+files[0])
    FaceFileName = "F:\\Aegis\\Capstone Project\\Capstone Project\\Final Project\\Final_Running_Code\\static\\sample_img\\"+files[0]
    path,dirs,files = os.walk('F:\\Aegis\\Capstone Project\\Capstone Project\\Final Project\\Final_Running_Code\\static\\sample_img').__next__()
    if(len(files)!=0):
        for f in files:
            os.remove(os.path.join(path,f))
    cv2.imwrite(FaceFileName, match_image)
    face_detector = dlib.get_frontal_face_detector()
    rects = face_detector(match_image, 1)
            
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = match_image[y: y + h, x: x + w]
        database.loc[index]=[crop_img,cap.get(cv2.CAP_PROP_POS_MSEC)]
    
    
    features_list= list()
    ## We need a 2048 feature vector, hence 224X224 target size of the image
    train_img = []
    for i in range(len(database)):
        temp_img = database.iloc[i,0]
        temp_img=cv2.resize(temp_img, (224, 224)) 
        temp_img=image.img_to_array(temp_img)
        features_list.append(database.iloc[i,0])
        train_img.append(temp_img)
     
    
    #converting train images to array and applying mean subtraction processing
    train_img=np.array(train_img)
    train_img=preprocess_input(train_img) 
    
    # loading VGG16 model weights, default pooling is none,could be made either avg or max.
    # include_top should be set to false, as this removes the fully connected output layer, and hence we only get feature vectors.
    #model = VGG16(weights='imagenet', include_top=False)
    model = VGGFace(model='resnet50',include_top=False)
    # Extracting features from the train dataset using the VGG16 pre-trained model
    
    features_train=model.predict(train_img)
    
    ## alternate to VGG is ResNet50
    #model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    
    ## t-SNE for Similarity Clustering
    from sklearn import decomposition,manifold,pipeline
    
    model = manifold.TSNE(random_state=0)
    
    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(features_train).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))
    
    # perform t-SNE
    vis_data = model.fit_transform(x_data)
    visdata=vis_data
    
    
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    def imscatter(x, y, image, ax=None, zoom=1):
        if ax is None:
            ax = plt.gca()
        try:
            image = plt.imread(image)
        except TypeError:
            # Likely already an array...
            pass
        im = OffsetImage(image, zoom=zoom)
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists 
    
    
#    import matplotlib.pyplot as plt
#    display= True
#    if display:
#        print("Plotting t-SNE visualization...")
#        fig, ax = plt.subplots()
#        imscatter(vis_data[:, 0], vis_data[:, 1], vis_data, ax=ax, zoom=0.15) 
#        plt.show()
#    
    
    
    class XMeans:
        def loglikelihood(self, r, rn, var, m, k):
            l1 = - rn / 2.0 * mt.log(2 * mt.pi)
            l2 = - rn * m / 2.0 * mt.log(var)
            l3 = - (rn - k) / 2.0
            l4 = rn * mt.log(rn)
            l5 = - rn * mt.log(r)
    
            return l1 + l2 + l3 + l4 + l5
    
        def __init__(self, X, kmax = 20):
            self.X = X
            self.num = np.size(self.X, axis=0)
            self.dim = np.size(X, axis=1)
            self.KMax = kmax
    
        def fit(self):
            k = 1
            X = self.X
            M = self.dim
            num = self.num
    
            while(1):
                ok = k
    
                #Improve Params
                kmeans = KMeans(n_clusters=k).fit(X)
                labels = kmeans.labels_
                m = kmeans.cluster_centers_
    
                #Improve Structure
                #Calculate BIC
                p = M + 1
    
                obic = np.zeros(k)
    
                for i in range(k):
                    rn = np.size(np.where(labels == i))
                    var = np.sum((X[labels == i] - m[i])**2)/float(rn - 1)
                    obic[i] = self.loglikelihood(rn, rn, var, M, 1) - p/2.0*mt.log(rn)
    
                #Split each cluster into two subclusters and calculate BIC of each splitted cluster
                sk = 2 #The number of subclusters
                nbic = np.zeros(k)
                addk = 0
    
                for i in range(k):
                    ci = X[labels == i]
                    r = np.size(np.where(labels == i))
    
                    kmeans = KMeans(n_clusters=sk).fit(ci)
                    ci_labels = kmeans.labels_
                    sm = kmeans.cluster_centers_
    
                    for l in range(sk):
                        rn = np.size(np.where(ci_labels == l))
                        var = np.sum((ci[ci_labels == l] - sm[l])**2)/float(rn - sk)
                        nbic[i] += self.loglikelihood(r, rn, var, M, sk)
    
                    p = sk * (M + 1)
                    nbic[i] -= p/2.0*mt.log(r)
    
                    if obic[i] < nbic[i]:
                        addk += 1
    
                k += addk
    
                # print(obic)
                # print(nbic)
    
                if ok == k or k >= self.KMax:
                    break
    
    
            #Calculate labels and centroids
            kmeans = KMeans(n_clusters=k).fit(X)
            self.labels = kmeans.labels_
            self.k = k
            self.m = kmeans.cluster_centers_
    
    
    #if __name__ == '__main__':

    xm = XMeans(visdata)
    xm.fit()
        
    
    new_data = pd.DataFrame(columns=['feat','X','Y','labels','vgg_extracted_feature','tsne_feature','time_stamp'])
    
    for i in range(len(vis_data)):
        new_data.loc[i]=[features_list[i],vis_data[i][0],vis_data[i][1],xm.labels[i],features_train[i],vis_data[i], database.iloc[i,1]]
      
    
            
    
    from scipy import spatial
    
    #original=new_data.iloc[(len(new_data)-1),0]
    original =np.hstack(np.hstack( new_data.iloc[(len(new_data)-1),4] ))
    s_list=list()
    for i in range(len(vis_data)-1):
        #contrast=np.hstack(new_data.iloc[i,0])
        contrast=np.hstack(np.hstack( new_data.iloc[i,4] ))
        s1=spatial.distance.euclidean(original, contrast)
        s_list.append(s1)
    
    
    #finding top 3 matched images
    greater= list()
    count =0
    index = np.arange(len(s_list))
    merged = sorted(zip(s_list,index ))
    for i in merged:
        count = count + 1
        if(count<=3):
            greater.append(i[1])
          
    
    #wrtiting the matched images to folder 
    #also creating dictionary with time stamp and immage link 
    top_images_dict={}
    def match_found():
        c=0
        for i in greater:
            c=c+1
            FaceFileName = "F:\\Aegis\\Capstone Project\\Capstone Project\\Final Project\\Final_Running_Code\\static\\results\\face_"+str(c)+".jpg"
            cv2.imwrite(FaceFileName, new_data.feat[i])
            millis = new_data.time_stamp[i]
            seconds=(millis/1000)%60
            seconds = int(seconds)
            minutes=(millis/(1000*60))%60
            minutes = int(minutes)
            hours=(millis/(1000*60*60))%24
            hours=int(hours)
            formated_time = str(hours)+':'+str(minutes)+':'+str(seconds)
            top_images_dict[formated_time] = "face_"+str(c)+".jpg"
    
    path,dirs,files = os.walk("F:\\Aegis\\Capstone Project\\Capstone Project\\Final Project\\Final_Running_Code\\static\\results").__next__()
    if(len(files)!=0):
        for f in files:
            os.remove(os.path.join(path,f))
        match_found()
    else:
        match_found()
    print("--- %s minutes ---" % ((time.time() - start_time)/60))    
    return top_images_dict
