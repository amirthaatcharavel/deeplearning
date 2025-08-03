import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')
    size = min(max(image.size), max_size)

    if shape:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize((size if isinstance(size, int) else shape)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),

        
                             (0.229, 0.224, 0.225))
    ])

    image = in_transform(image).unsqueeze(0)
    return image

# Function to convert a tensor to a displayable image
def im_convert(tensor):
    image = tensor.clone().detach().squeeze(0)
    image = image.to('cpu').numpy().transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    return image

# Load content and style images
content = load_image("content.jpg")
style = load_image("cpainting.jpg", shape=content.shape[-2:])

# Load the pretrained VGG19 model
vgg = models.vgg19(pretrained=True).features

# Freeze all VGG parameters
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)
content = content.to(device)
style = style.to(device)

# Layers to be used for style and content representation
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content layer
            '28': 'conv5_1'
        }
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

# Gram matrix for style representation
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Get features of content and style
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Compute Gram matrices for style features
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Initialize the target image
target = content.clone().requires_grad_(True).to(device)

# Define weights for style layers
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}

content_weight = 1e4
style_weight = 1e2

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Style Transfer Loop
steps = 300
for step in range(steps):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        _, d, h, w = target_feature.shape
        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total loss: {total_loss.item():.4f}")

# Save the result
final_img = im_convert(target)
plt.imsave("stylized_output.jpg", final_img)
print("✅ Stylized image saved as stylized_output.jpg")