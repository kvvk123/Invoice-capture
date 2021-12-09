import os
from flask import Flask,request,redirect,url_for,render_template
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import label_map_util
import visualization_utils as vis_util
import string_int_label_map_pb2
import os
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from run_interface import run_inference_for_single_image
from csv_writer import csv_writer
import collections
import re
import pytesseract
import pandas as pd
import csv
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


sys.path.append("..")
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
NUM_CLASSES = 6

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

app = Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape( (im_height, im_width, 3)).astype(np.uint8)


def convertImage(image_path):
    print(image_path)
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    coordinates = vis_util.return_coordinates(
                    image_np,
                    np.squeeze(output_dict['detection_boxes']),
                    np.squeeze(output_dict['detection_classes']).astype(np.int32),
                    np.squeeze(output_dict['detection_scores']),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.60)
    structuredFormat ={}
    structuredFormat['Amount'] = ''
    structuredFormat['Items'] =''
    structuredFormat['InvoiceNumber']=''
    structuredFormat['ClientInfo'] =''
    structuredFormat['CompanyInfo']=''
    structuredFormat['InvoiceDate'] =''

    for cor in coordinates:
        (y1, y2, x1, x2, acc,classes) = cor
        newvalues = re.sub(r'[^A-Za-z]','',acc)
        # print(acc)
        if newvalues == 'Items': 
            height = y2-y1
            width = x2-x1
            crop = image_np[y1:y1+height, x1:x1+width]
            text = pytesseract.image_to_string(crop)
            text = text.replace('\n','')
            structuredFormat['Items'] = text.replace('\n','')
        elif newvalues == 'CompanyInfo':
            height = y2-y1
            width = x2-x1

            crop = image_np[y1:y1+height, x1:x1+width]
            text = pytesseract.image_to_string(crop)
            text = text.replace('\n','')
            structuredFormat['CompanyInfo'] = text.replace('\n','')

        elif newvalues == 'ClientInfo':
            height = y2-y1
            width = x2-x1

            crop = image_np[y1:y1+height, x1:x1+width]
            text = pytesseract.image_to_string(crop)
            text = text.replace('\n','')
            structuredFormat['ClientInfo'] = text.replace('\n','')

        elif newvalues == 'InvoiceNumber':
            height = y2-y1
            width = x2-x1

            crop = image_np[y1:y1+height, x1:x1+width]
            text = pytesseract.image_to_string(crop)
            structuredFormat['InvoiceNumber'] = text.replace('\n','')

        elif newvalues == 'InvoiceDate':
            height = y2-y1
            width = x2-x1

            crop = image_np[y1:y1+height, x1:x1+width]
            text = pytesseract.image_to_string(crop)
            structuredFormat['InvoiceDate'] = text.replace('\n','')

        elif newvalues == 'Amount':
            height = y2-y1
            width = x2-x1

            crop = image_np[y1:y1+height, x1:x1+width]
            text = pytesseract.image_to_string(crop)
            structuredFormat['Amount'] = text.replace('\n','')
            print(structuredFormat)
    return structuredFormat

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=="POST":
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(file_path)
            result = convertImage(file_path) 
            print(result)
            with open('static/result.csv', 'w') as f:
                for key in result.keys():
                    f.write("%s,%s\n"%(key,result[key]))
            return render_template('index.html', image="../static/"+filename, result=result)
    return render_template('index.html', image="", result="")


if __name__ =='__main__':
    app.run(debug=True)