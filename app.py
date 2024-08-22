import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
import os

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

app = Flask(__name__)

#trained fonts
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

model = PrimaryModel()
model.load_state_dict(torch.load("primary_model_final/best_model",map_location=torch.device('cpu')))

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)


        model = PrimaryModel()
        model.load_state_dict(torch.load("primary_model_final/best_model", map_location=torch.device('cpu')))
        model.eval()

   
        filepath = 'static/uploads/Scan1.png'


        img = Image.open(filepath).convert('RGB')  
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor(),                     
            transforms.Normalize((0.5,), (0.5,))  
        ])


        img = transform(img).unsqueeze(0)  


        with torch.no_grad():  
            prediction = model(img)
        outputs_np = prediction.detach().numpy()
        best3ind = np.argsort(outputs_np,axis=1)[:,-3:][:, ::-1] 
        best3prob = np.sort(outputs_np,axis=1)[:,-3:][:, ::-1]
        output = [['%s, confidence: %s'%(classes[best3ind[0][j]], round(best3prob[0][j]*100,2))+'%' for j in range(3)]]


        return render_template('index.html', filename=filename, output=output)

if __name__ == "__main__":
    app.run(debug=True)
