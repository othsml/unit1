# PyTorch CNN: Core Training Logic (Minimalist)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import requests 
import io
from PIL import Image

# --- 1. The Model Architecture ---
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) 
        # Fully Connected Layers (calculated input size: 16 * 5 * 5 = 400)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 outputs for CIFAR-10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

net = Net()

# --- 2. Data Setup & Loading ---
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load training data (requires internet)
print("Loading CIFAR-10 data...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- 3. Training ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Starting Training (20 Epochs)...")
num_epochs = 20

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 5000 == 4999:    
            print(f'[E{epoch + 1}] Loss: {running_loss / 5000:.3f}')
            running_loss = 0.0
            
print('Training finished.')

# --- 4. Custom Image Prediction ---
CUSTOM_IMAGE_URL = "https://www.topgear.com/sites/default/files/images/news-article/2017/03/9d77d8a226a932a0def035bc4892eaab/koenigseggagerarsgeneva2017-1.jpg"

# Fetch, preprocess
print("\nPredicting custom image...")

# Added simple headers to mimic a browser and ensure a direct image file is returned
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
# Removed stream=True and added headers and allow_redirects=True for robustness
response = requests.get(CUSTOM_IMAGE_URL, headers=headers, allow_redirects=True) 

image = Image.open(io.BytesIO(response.content)).convert('RGB')
input_tensor = transform(image).unsqueeze(0) 

# Run prediction
with torch.no_grad():
    net.eval()
    outputs = net(input_tensor)
    
# Interpret results
probabilities = F.softmax(outputs, dim=1)
predicted_prob, predicted_index = torch.max(probabilities, 1)
predicted_class = classes[predicted_index.item()]

print("--------------------------------------")
print(f"Prediction: {predicted_class.upper()}")
print(f"Confidence: {predicted_prob.item()*100:.2f}%")