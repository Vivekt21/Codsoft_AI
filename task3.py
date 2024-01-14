import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import json

# Download NLTK data for tokenization
nltk.download('punkt')

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image captioning model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

# Load pre-trained ResNet model for image feature extraction
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)
resnet.eval()

# Load pre-trained tokenizer for captions
with open('tokenizer.json', 'r') as f:
    tokenizer = json.load(f)
vocab_size = len(tokenizer) + 1

# Create the image captioning model
embed_size = 256
hidden_size = 512
num_layers = 1
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers).to(device)

# Load pre-trained weights for the image captioning model
model.load_state_dict(torch.load('image_captioning_model.pth'))
model.eval()

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Generate captions for the input image
def generate_caption(image_path, model, tokenizer, max_length=20):
    model.eval()
    with torch.no_grad():
        image = preprocess_image(image_path)
        features = resnet(image).squeeze(2).squeeze(2)
        features = features.unsqueeze(1)

        captions = ['<start>']
        for _ in range(max_length):
            input_captions = [tokenizer[word] for word in captions]
            input_captions = torch.LongTensor(input_captions).unsqueeze(0).to(device)
            lengths = [len(captions)]

            outputs = model(features, input_captions, lengths)
            predicted_word_index = outputs.argmax(2)[:, -1].item()
            predicted_word = [word for word, index in tokenizer.items() if index == predicted_word_index][0]
            captions.append(predicted_word)

            if predicted_word == '<end>':
                break

    generated_caption = ' '.join(captions[1:-1])
    return generated_caption

# Example usage
image_path = 'ss.jpg'
caption = generate_caption(image_path, model, tokenizer)
print(f"Generated Caption: {caption}")
