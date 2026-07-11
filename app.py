import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 1. Set page configurations
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

# 2. Add some premium custom CSS for styling
st.markdown("""
    <style>
    .title-text {
        text-align: center;
        color: #1E3D59;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
    }
    .subtitle-text {
        text-align: center;
        color: #17B978;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title-text">🐱 Cat vs Dog Image Classifier 🐶</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload an image of a cat or dog to see its breed classification</p>', unsafe_allow_html=True)

# 3. Load the model (cached so it only loads once)
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()
    categories = models.MobileNet_V2_Weights.DEFAULT.meta["categories"]
    return model, categories

try:
    model, categories = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# 4. Define Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 5. File Uploader UI
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        img = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            with st.spinner("Analyzing image..."):
                tensor = preprocess(img).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # ImageNet category index slices for Cats and Dogs
                cat_prob = float(torch.sum(probabilities[281:294]))
                dog_prob = float(torch.sum(probabilities[151:276]))
                
                top_prob, top_idx = torch.max(probabilities, dim=0)
                top_class = categories[int(top_idx)]
                
                st.markdown("### Results")
                
                # Determine classification
                if (cat_prob + dog_prob) < 0.18:
                    st.warning("⚠️ **Classification: Neither / Other**")
                    st.info(f"Top class detected: **{top_class.replace('_', ' ').title()}** ({top_prob*100:.1f}%)")
                elif cat_prob > dog_prob:
                    st.success("🐱 **Classification: CAT**")
                    cat_probs = probabilities[281:294]
                    top_cat_idx = int(torch.argmax(cat_probs)) + 281
                    breed = categories[top_cat_idx]
                    st.write(f"Detected Breed: **{breed.replace('_', ' ').title()}**")
                else:
                    st.success("🐶 **Classification: DOG**")
                    dog_probs = probabilities[151:276]
                    top_dog_idx = int(torch.argmax(dog_probs)) + 151
                    breed = categories[top_dog_idx]
                    st.write(f"Detected Breed: **{breed.replace('_', ' ').title()}**")
                
                # Confidence Breakdown
                st.markdown("---")
                st.write("**Confidence Breakdown:**")
                st.progress(cat_prob, text=f"Cat Probability: {cat_prob*100:.1f}%")
                st.progress(dog_prob, text=f"Dog Probability: {dog_prob*100:.1f}%")
                
    except Exception as e:
        st.error(f"Error classifying the image: {e}")
