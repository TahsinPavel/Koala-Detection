from flask import Flask, render_template, request, session, Response

import os
import cv2
from werkzeug.utils import secure_filename
 
import cv2
import numpy as np
 
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'You Will Never Guess'
 
#  object detection function
def detect_object(uploaded_image_path):
    # Loading image
    img = cv2.imread(uploaded_image_path)
 
    # Load Yolo
    weight = "weights\yolov3-custom_2000.weights"
    config = "cfg\yolov3-custom.cfg"
    label = "data\labels\custom.names"
    net = cv2.dnn.readNet(weight, config)
 
    classes = []
    with open(label, "r") as f:
        classes = [line.strip() for line in f.readlines()]
 
    # print(classes)
 
    # # Defining desired shape
    fWidth = 320
    fHeight = 320
 
    # Resize image in opencv
    img = cv2.resize(img, (fWidth, fHeight))
 
    height, width, channels = img.shape
 
    # Convert image to Blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (fWidth, fHeight), (0, 0, 0), True, crop=False)
    # Set input for YOLO object detection
    net.setInput(blob)
 
    # Find names of all layers
    layer_names = net.getLayerNames()
    # print(layer_names)
    # Find names of three output layers
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print(output_layers)
 
    # Send blob data to forward pass
    outs = net.forward(output_layers)
    print(outs[0].shape)
    print(outs[1].shape)
    print(outs[2].shape)
 
    # Generating random color for all 80 classes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
 
    # Extract information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            # Extract score value
            scores = detection[5:]
            # Object id
            class_id = np.argmax(scores)
            # Confidence score for each object ID
            confidence = scores[class_id]
            # if confidence > 0.5 and class_id == 0:
            if confidence > 0.5:
                # Extract values to draw bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
 
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_label = int(confidences[i] * 100)
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{label, confidence_label}', (x - 25, y + 75), font, 1, color, 2)
 
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.jpg')
    cv2.imwrite(output_image_path, img)
 
    return(output_image_path)
 
 
@app.route('/')
def index():
    return render_template('upload_img.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
 
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
 
        return render_template('submit_img.html')
 
@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    return render_template('show_image.html', user_image = img_file_path)
 
@app.route('/detect_object')
def detectObject():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    output_image_path = detect_object(uploaded_image_path)
    print(output_image_path)
    return render_template('show_image.html', user_image = output_image_path)
 
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
 
if __name__=='__main__':
    app.run(debug = True)