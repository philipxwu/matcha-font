import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from waitress import serve
import matplotlib.pyplot as plt
from PIL import Image
from werkzeug.utils import secure_filename

classes = ['Candara', 'Deng', 'Gabriola', 'HPSimplified', 'Inkfree', 'Montserrat-Regular', 'SitkaVF', 'arial', 'bahnschrift', 
 'calibri', 'cambriab', 'comic', 'consola', 'constan', 'corbel', 'cour', 'ebrima', 'framd', 'georgia', 'himalaya', 
 'impact', 'javatext', 'l_10646', 'lucon', 'malgun', 'micross', 'monbaiti', 'msyi', 'mvboli', 'pala', 'phagspa', 
 'segoepr', 'segoesc', 'seguili', 'simfang', 'simkai', 'sylfaen', 'symbol', 'tahoma', 'times', 'trebuc', 'verdana']

#model declaration 
class PrimaryModel(nn.Module):
    def __init__(self):
        super(PrimaryModel, self).__init__() 
        self.conv1 = nn.Conv2d(1, 16, 3, 2) 
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, 1)
        self.bn4 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*27*27, 32*32)
        self.bn5 = nn.BatchNorm1d(32*32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32*32, 42)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        
        x = x.view(-1, 32*27*27)
        
        x = self.bn5(self.dropout(F.relu(self.fc1(x))))
        x = F.softmax(self.fc2(x), dim=1)

        return x
    
def expand2square(pil_img):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(mode = 'RGB',size= (width, width),color = (255,255,255))
        result.paste(pil_img, (0, (width - height) // 2))
        return result.resize((224,224))
    else:
        result = Image.new(mode = 'RGB', size = (height, height), color = (255,255,255))
        result.paste(pil_img, ((height - width) // 2, 0))
        return result.resize((224,224))

UPLOAD_FOLDER = 'static/uploads/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
@app.route('/index')

def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def result():

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
    
    img = Image.open(filepath).convert('RGB')  
    
    img = expand2square(img)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(),                     
        transforms.Normalize((0.5,), (0.5,))  
    ])
    img = transform(img).unsqueeze(0)  
    
    model = PrimaryModel()
    model.load_state_dict(torch.load("primary_model_final/best_model", map_location=torch.device('cpu')))
    model.eval()
    
    


    with torch.no_grad():  
        prediction = model(img)
    outputs_np = prediction.detach().numpy()
    best3ind = np.argsort(outputs_np,axis=1)[:,-3:][:, ::-1] 
    best3prob = np.sort(outputs_np,axis=1)[:,-3:][:, ::-1]
    outputname = classes[best3ind[0][0]]
    outputpercent = round(best3prob[0][0]*100,2)

    folder_dir = 'static/fonts_image_dataset/' + outputname
    for imagesfont in os.listdir(folder_dir):
        folderimg = outputname + '/'+ imagesfont

                

    return render_template('result.html', outputname = outputname, filename= filename, outputpercent = outputpercent, folderimg = folderimg)

    

if __name__ == "__main__":
    serve(app, host = "0.0.0.0", port=8000)
