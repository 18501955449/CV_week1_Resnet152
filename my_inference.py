'''自己写resnet152测试'''
import torch
from resnet152 import resnet152
from torchvision import transforms
from PIL import Image
file = "resnet152-b121ed2d.pth"
model = resnet152(pretrained=False, progress=True)
model.load_state_dict(torch.load(file))
model.eval()

def predict(model):
    filename = 'dog.jpg'
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    probability = torch.nn.functional.softmax(output,dim=1)
    max_value, index = torch.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
    print(max_value,index)
predict(model)