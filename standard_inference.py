'''官方resnet152测试代码'''
import torch
from torchvision import models
from PIL import Image
from torchvision import transforms
model = models.resnet152(pretrained=False)
model.load_state_dict(torch.load('resnet152-b121ed2d.pth'))
model.eval()
def predict(model):
    filename = 'dog.jpg'
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    #print(torch.nn.functional.softmax(output[0], dim=0))
    probability = torch.nn.functional.softmax(output,dim=1)
    max_value, index = torch.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
    return index

index = predict(model)
print(index)