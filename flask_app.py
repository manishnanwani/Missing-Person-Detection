    # -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:27:27 2018

"""
#%%
from Intelligent_Vision import *
import os
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename


UPLOAD_FOLDER_Video = 'F:\\Aegis\\Capstone Project\\Capstone Project\\Final Project\\Final_Running_Code\\uploads\\new_video'
UPLOAD_FOLDER_Image = 'F:\\Aegis\\Capstone Project\\Capstone Project\\Final Project\\Final_Running_Code\\uploads\\new_image'

 #UPLOAD_FOLDER_Static_Image= 'F:\\Aegis\\Capstone Project\\Capstone Project\\Final Project\\Final_Running_Code\\static\\sample_img'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','mp4','3gp'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER_Video'] = UPLOAD_FOLDER_Video
app.config['UPLOAD_FOLDER_Image'] = UPLOAD_FOLDER_Image
#app.config['UPLOAD_FOLDER_Static_Image'] = UPLOAD_FOLDER_Static_Image
#%%
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
#        if 'file' not in request.files:
#            flash('No file part')
#            return redirect(request.url)
        video_file = request.files['file_video']
        # if user does not select file, browser also
        # submit an empty part without filename
        video_upload_status=False
        img_upload_status=False
        if video_file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if video_file and allowed_file(video_file.filename):
            filename = secure_filename(video_file.filename)
            video_upload_status=True
            video_file.save(os.path.join(app.config['UPLOAD_FOLDER_Video'], filename))

        img_file = request.files['file_image']
        # if user does not select file, browser also
        # submit an empty part without filename
        if img_file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            img_upload_status=True
            img_file.save(os.path.join(app.config['UPLOAD_FOLDER_Image'], filename))
#            Also write to static folder for reference.
            #img_file.save(os.path.join(app.config['UPLOAD_FOLDER_Static_Image'], filename))
        if(video_upload_status and img_upload_status):
            #Perform Matching
            return render_template(
                'results.html', results=detect_faces_in_image(video_file,img_file),sample=str(img_file.filename)
        
            )
        elif(not video_upload_status):
            return '''<h1> Error in Uploading video</h1>'''
        elif(not img_upload_status):
           return '''<h1> Error in Uploading img</h1>'''

    return render_template('index.html')

def detect_faces_in_image(video_file,img_file):
    print(" Files Currently Uploaded are : ",str(video_file.filename),"\t",str(img_file.filename))
    results_matched = Intelligent_Vision()
    #Run your matching function
    #results_matched={1:'face_1.jpg',2:'face_2.jpg',3:'face_3.jpg'}
    return results_matched 

if  __name__ =="__main__":
    app.debug = True
    app.secret_key='manish'
    app.config['SESSION_TYPE']='filesystem'
    path,dirs,files = os.walk( UPLOAD_FOLDER_Video).__next__()
    if(len(files)!=0):
        for f in files:
            os.remove(os.path.join(path,f))
    path,dirs,files = os.walk( UPLOAD_FOLDER_Image).__next__()
    if(len(files)!=0):
        for f in files:
            os.remove(os.path.join(path,f))
    app.run()
