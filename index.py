import mimetypes
import os
from urllib.parse import unquote_plus
from flask import Flask, send_file, send_from_directory
from flask import request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math
from flask import Flask, request
from flask_restful import Api, Resource
from flask import Flask, jsonify
from flask_cors import CORS
from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math
from model import YoloEffect
from model import delete_image_files
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from flask_mysqldb import MySQL 
import urllib
import pyrebase
from urllib.parse import unquote_plus
from urllib.parse import urlparse, unquote
import urllib.parse
from model import convert_avi_to_mp4

app = Flask(__name__)
api = Api(app)
CORS(app) 
app = Flask(__name__)


firebaseConfig = {
  "apiKey": "AIzaSyDTzQ1MCG_OJPdQGcwfc0OnLEzakcpzjEQ",
  "authDomain": "prompt-397314.firebaseapp.com",
  "projectId": "prompt-397314",
  "storageBucket": "prompt-397314.appspot.com",
  "messagingSenderId": "885517242235",
  "appId": "1:885517242235:web:2f509be14004da261677de",
  "measurementId": "G-0PW0HM9KT0",
  # "databaseURL": "https://prompt-397314.firebaseio.com"
  "databaseURL": "https://prompt-397314-default-rtdb.firebaseio.com/"
}

firebase=pyrebase.initialize_app(firebaseConfig)
storage=firebase.storage()

#convert file url to filename
def extract_filename_from_url(url):
    try:
        # Parse the URL to extract the path and query components
        parsed_url = urllib.parse.urlparse(url)

        # Extract the path component from the parsed URL
        path = parsed_url.path
        
        # Split the path by '/' to find the parts containing the filename
        path_parts = path.split('/')
        
        # Find the index of 'o' in the path parts
        try:
            index_of_o = path_parts.index('o')
        except ValueError:
            return None  # 'o' not found in the path, invalid URL format
        
        # Extract the file path parts after 'o/'
        file_path_parts = path_parts[index_of_o + 1:]
        
        # Join the file path parts into a string
        file_path = '/'.join(file_path_parts)
        
        # Decode URL-encoded characters in the file path
        file_path_decoded = urllib.parse.unquote(file_path)
        
        # Split the decoded file path by '?' to remove query parameters
        file_path_cleaned = file_path_decoded.split('?')[0]
        
        # Extract the filename from the cleaned file path
        filename = file_path_cleaned.split('/')[-1]
        
        return filename
    except Exception as e:
        print(f"Error extracting filename from URL '{url}': {str(e)}")
        return None




#delete from firebase
def delete_file_by_filename(filename):
    try:
        # Construct the full path to the file in Firebase Storage
        file_path = "media/media/" + filename
        # Get a reference to the file in Firebase Storage
        file_ref = storage.child(file_path)
        if file_ref.get_url(None):
            # Delete the file
            file_ref.delete(file_path,None)
        
        print(f"File '{filename}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting file '{filename}': {str(e)}")





#loob on images and delete them from firebase storage
def delete_from_storage(url_arr):
    for url in url_arr:
        filename = extract_filename_from_url(url)
        delete_file_by_filename(filename)



def store_to_database(names):
    url_arr = []
    for _ in names:
        fle = _
        cloudfilename=f"media/{fle}"
        storage.child(cloudfilename).put(fle)
        url_arr.append(storage.child(cloudfilename).get_url(None))

    # delete_from_storage(url_arr)
 
    return url_arr
        






@app.route('/upload_image', methods=['POST'])
@cross_origin(origin="http://localhost:3000")
def upload_image():
    cls = YoloEffect()
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        randname = cls.generate_randomname() + ".jpg"
        file.filename = randname
        file.save(os.path.join('', file.filename))
        names = []
        
        ret , clss, sizes= cls.cut(file.filename)
        for _ in ret: names.append(_)
        ret = cls.redirect(ret)
        # os.path.dirname(os.path.abspath(__file__))
        
        #GANs
        
       
        
        
        return jsonify({'message':ret,'classes': clss,'sizes': sizes ,'status':200})
    except Exception as e:
        return jsonify({'error': f'Failed to save file - {str(e)}'}), 500
    
    
    

@app.route('/upload_video', methods=['POST'])
@cross_origin(origin="http://localhost:3000")
def upload_video():
    cls = YoloEffect()
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        cls = YoloEffect() # to use segment function
        randname = cls.generate_randomname() + ".mp4" # to generate name
        file.filename = randname  # rename the file
        file.save(os.path.join('', file.filename)) # save the file beside this file
        ret , size = cls.segment(file.filename)
        return jsonify({"ret": ret, 'size':size,'status':200})
    except Exception as e:
        return jsonify({'error': f'Failed to save file - {str(e)}'}), 500

 


@app.route('/delete', methods=['POST'])
@cross_origin(origin="http://localhost:3000")
def delete_images():
    if request.method == 'POST':
        data = request.json.get('data') 
        print(data)
        arr = []
        for filename in data:
            idx = filename.find("media")
            arr.append(filename[idx:])
        delete_image_files(arr)
        return 'deleted successfully', 200
    
    
#api to delete from firebase 
@app.route('/deletefb', methods=['POST'])
@cross_origin(origin="http://localhost:3000")
def delete_images_from_firebase():
    if request.method == 'POST':
        data = request.json.get('data') 
        delete_from_storage(data)
 
        return 'deleted successfully', 200
    
    
    
@app.route('/store', methods=['POST'])
@cross_origin(origin="http://localhost:3000")
def store():
    if request.method == 'POST':
        data = request.json.get('data') 
        try:
            a = []
            for filename in data:
                idx = filename.find("media")
                a.append(filename[idx:])
            url = store_to_database(a)
            delete_image_files(a)
            # delete_from_storage(url)
            return jsonify({'images_url':url, 'status':200})
        except:
            return 'error', 500
            
        
        
        


@app.route('/media/<path:filename>')
def serve_media(filename):
    media_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')
    if os.path.isfile(os.path.join(media_dir, filename)):
        return send_from_directory(media_dir, filename)
    else:
        os.abort(404) 
    
@app.route('/')
def hello():
    return f'Hi Abdul'

if __name__ == '__main__':
    app.run(port=5000, debug=True)
    
    
    
    