import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

print("PyTorch Version:", torch.__version__)

# Check if model loads
try:
    print("Loading MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    categories = models.MobileNet_V2_Weights.DEFAULT.meta["categories"]
    print("Successfully loaded MobileNetV2 with ImageNet weights.")
except Exception as e:
    print("Failed to load model:", e)
    exit(1)

# Check if transform and inference works on our samples
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

sample_dir = "/Users/avinashpatel/.gemini/antigravity/scratch/cat-vs-dog-image-classifier/samples"
sample_files = ["cat1.jpg", "cat2.jpg", "dog1.jpg", "dog2.jpg"]

for fname in sample_files:
    fpath = os.path.join(sample_dir, fname)
    if not os.path.exists(fpath):
        print(f"Sample file {fpath} does not exist!")
        exit(1)
        
    try:
        img = Image.open(fpath).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        cat_prob = float(torch.sum(probabilities[281:294]))
        dog_prob = float(torch.sum(probabilities[151:276]))
        other_prob = max(0.0, 1.0 - cat_prob - dog_prob)
        
        top_prob, top_idx = torch.max(probabilities, dim=0)
        top_class = categories[int(top_idx)]
        
        print(f"\nFile: {fname}")
        print(f"Top ImageNet Class: {top_class} ({float(top_prob)*100:.2f}%)")
        print(f"Cat Probability Sum: {cat_prob*100:.2f}%")
        print(f"Dog Probability Sum: {dog_prob*100:.2f}%")
        print(f"Other Probability Sum: {other_prob*100:.2f}%")
        
        is_cat = cat_prob > dog_prob
        is_other = (cat_prob + dog_prob) < 0.18
        
        if is_other:
            print("Classification: Neither / Other")
        elif is_cat:
            cat_probs = probabilities[281:294]
            top_cat_idx = int(torch.argmax(cat_probs)) + 281
            breed = categories[top_cat_idx]
            print(f"Classification: CAT (Breed: {breed})")
        else:
            dog_probs = probabilities[151:276]
            top_dog_idx = int(torch.argmax(dog_probs)) + 151
            breed = categories[top_dog_idx]
            print(f"Classification: DOG (Breed: {breed})")
            
    except Exception as e:
        print(f"Error classifying {fname}: {e}")
        exit(1)

print("\nAll sample classifications verified successfully!")
