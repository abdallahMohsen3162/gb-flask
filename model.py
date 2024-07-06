import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import math
import cv2
import random
import pyrebase
import os
from segmentv import Segment_video
import shutil
from moviepy.editor import VideoFileClip
from eval import Generate_image
from eval import process_video

def convert_avi_to_mp4(input_path, output_path):
    try:
        # Load the AVI file
        clip = VideoFileClip(input_path)
        # Define the output file path
        output_file = output_path
        # Save the clip in MP4 format
        clip.write_videofile(output_file, codec='libx264')
    except Exception as e:
        print(f"Error converting video: {e}")


# hosting = "http://127.0.0.1:8080/"

hosting = "http://127.0.0.1:5000/"

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


media_folder = "media/"

allowed_classes = ["person", "original", "car", "truck"]
def delete_image_files(file_paths):
    for path in file_paths:
        try:
            os.remove(path)
        except OSError as e:
            print(f"Error deleting {path}: {e}")

# deletes the hosting :
# http://127.0.0.1:8080/media/383890651303200.mp4 => media/383890651303200.mp4
def preproccess(data):
    a = []
    for filename in data:
        idx = filename.find("media")
        a.append(filename[idx:])
        
    return a

class YoloEffect:
 def __init__(self):
  pass
  
 def generate_randomname(self):
    ret = ""

    for _ in range(14):
        ret += str(random.randint(0, 9))
    return ret

 def redirect(self,ret):
    for i in range(len(ret)):
        ret[i] = hosting + ret[i]
    return ret


 def segment(self, imgname, low_res):
  ret = []
  
  outputname = self.generate_randomname() + ".avi"
  outputnamempavi = self.generate_randomname() + ".avi"
  outputnamemp4Name = self.generate_randomname() + ".mp4"
  #segmen video and save it as .avi
  Segment_video(imgname, outputname)
  process_video(outputname, outputnamempavi,10,low_res)
  #convert to .mp4
  convert_avi_to_mp4(outputnamempavi, media_folder+outputnamemp4Name)
  
  size = [os.path.getsize(media_folder+outputnamemp4Name)]
  ret.append(media_folder+outputnamemp4Name)
  #delete the main video
      
  file_path = os.path.join('', outputname) 
  if os.path.exists(file_path):
      os.remove(file_path)
      
      
  file_path = os.path.join('', imgname)
  if os.path.exists(file_path):
      os.remove(file_path)
        
    # Delete the intermediate .avi files
  file_path = os.path.join('', outputname)
  if os.path.exists(file_path):
      os.remove(file_path)
        
  file_path = os.path.join('', outputnamempavi)
  if os.path.exists(file_path):
      os.remove(file_path)
      
  return self.redirect(ret),size




 def cut(self, imgname, low_res, Yolo = False):
  ret = []  
  clss = ["original"]
  
  model = YOLO("yolov5lu.pt")
  res = model(imgname, show=False)
  cv2.waitKey(0)
  c = 0
  image = cv2.imread(imgname)
  img_name = self.generate_randomname()
  file_path = os.path.join(media_folder, f"{img_name}{c}.jpg")
  cv2.imwrite(file_path, image)

  ret.append(media_folder+img_name + f"{c}" + ".jpg" )
  print("Yolo = ",Yolo)
  if Yolo == "true":
      for i in res[0].boxes.data:
          print("############################")
          print("lowres = ", low_res)
          c += 1
          original_image = cv2.imread(imgname)
          if classNames[int(res[0].boxes.cls[c - 1])] not in allowed_classes: continue
          x, y, w, h = math.floor(i[0]), math.floor(i[1]), math.floor(i[2]), math.floor(i[3])
          cropped_image = original_image[y:h, x:w]
          img_name = self.generate_randomname()      
          file_path = os.path.join(media_folder, f"{img_name}{c}.jpg")
          cv2.imwrite(file_path, cropped_image)
          ret.append(media_folder+img_name + f"{c}" + ".jpg")
          clss.append(classNames[int(res[0].boxes.cls[c - 1])])
  
  file_path = os.path.join('', imgname) 

  if os.path.exists(file_path):
      os.remove(file_path)
  print("after loop")
  retbefore = []
  for i in range(len(ret)):
      print("cur = ",ret[i])
      retbefore.append(ret[i])
      newname = media_folder+self.generate_randomname()+".jpg"
      Generate_image(ret[i], newname, low_res)
    #   os.remove(ret[i])
      ret[i] = newname
      
  sizes = [os.path.getsize(i) for i in ret]
  return ret, clss, sizes, retbefore
  