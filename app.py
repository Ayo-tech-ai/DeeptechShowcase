import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os
from transformers import pipeline
import time

# -------------------------------
# UI Title & Description
# -------------------------------
st.set_page_config(page_title="AI-Powered LeafScan Assistant", layout="centered")
st.title("🌾 AI-Powered LeafScan Assistant")
st.markdown("*An AI assistant for detecting rice leaf diseases and providing expert guidance*")

# -------------------------------
# Sidebar (Optional Info)
# -------------------------------
with st.sidebar:
    st.header("About")
    st.write("This app uses a Convolutional Neural Network (CNN) to classify rice leaf diseases and a transformer-based NLP model to answer user questions.")
    st.markdown("**Built for:** 3MTT Monthly Showcase")
    st.markdown("**Creator:** Ayoola Mujib Ayodele")
    st.markdown("[GitHub Repo](https://github.com/Ayo-tech-ai/DeeptechShowcase)")

# -------------------------------
# 1. Load CNN Model from Google Drive
# -------------------------------

MODEL_PATH = "RiceClassifier.pth"
FILE_ID = "13nlieOIczZPmbCaA8M2AlefOrXINTXyL"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.eval()
    return model

cnn_model = load_cnn_model()

# -------------------------------
# 2. Class Names
# -------------------------------

class_names = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bakanae',
    'brown_spot', 'grassy_stunt_virus', 'healthy_rice_plant',
    'narrow_brown_spot', 'ragged_stunt_virus', 'rice_blast',
    'rice_false_smut', 'sheath_blight', 'sheath_rot',
    'stem_rot', 'tungro_virus'
]

# -------------------------------
# 3. NLP Q&A Model
# -------------------------------

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

qa_pipeline = load_qa_model()

# -------------------------------
# 4. Disease-Specific Knowledge Base
# -------------------------------

context_map = {
    "rice_blast": """Rice Blast, caused by the fungus Magnaporthe oryzae, ...""",
    "bacterial_leaf_blight": """Bacterial Leaf Blight is caused by Xanthomonas oryzae pv. oryzae. ...""",
    "bacterial_leaf_streak": """Bacterial Leaf Streak is caused by Xanthomonas oryzae pv. oryzicola. ...""",
    "bakanae": """Bakanae Disease, caused by Fusarium fujikuroi, ...""",
    "brown_spot": """Brown Spot is caused by Bipolaris oryzae. ...""",
    "grassy_stunt_virus": """Rice Grassy Stunt Virus (RGSV) leads to pale leaves and stunted growth. ...""",
    "narrow_brown_spot": """Caused by Cercospora janseana, this disease shows up as narrow streaks. ...""",
    "ragged_stunt_virus": """Rice Ragged Stunt Virus causes ragged leaves and stunted plants. ...""",
    "rice_false_smut": """False Smut forms black-green spore balls. ...""",
    "sheath_blight": """Caused by Rhizoctonia solani. Produces sheath lesions. ...""",
    "sheath_rot": """Sheath Rot causes grain shriveling and rot. ...""",
    "stem_rot": """Stem Rot involves black sclerotia in the stem. ...""",
    "tungro_virus": """Tungro is caused by two viruses and spread by green leafhoppers. ...""",
    "healthy_rice_plant": """This plant appears to be healthy. No treatment is required. ..."""
}

# -------------------------------
# 5. Image Preprocessing
# -------------------------------

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# -------------------------------
# Step 1: Upload + Diagnose
# -------------------------------

st.markdown("## Step 1: Upload & Diagnose")

uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"], key="uploaded_file")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Disease"):
        with st.spinner("Classifying..."):
            input_tensor = preprocess_image(image)
            outputs = cnn_model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

            label = class_names[predicted_class.item()]
            conf = confidence.item() * 100

            st.session_state["predicted_label"] = label
            st.session_state["confidence"] = conf

# Show CNN result
if "predicted_label" in st.session_state:
    st.success(f"**Disease Predicted:** {st.session_state.predicted_label}")
    st.info(f"**Confidence Level:** {st.session_state.confidence:.2f}%")

# -------------------------------
# Step 2: Ask the Assistant
# -------------------------------

st.markdown("## Step 2: Ask the Smart Assistant")

with st.form(key="qa_form"):
    question = st.text_input("Ask a question about rice diseases (e.g. How do I treat rice blast?)")
    submit = st.form_submit_button("Submit")

    if submit and question:
        with st.spinner("Thinking..."):
            disease_key = (
                st.session_state["predicted_label"].lower().replace(" ", "_")
                if "predicted_label" in st.session_state
                else None
            )
            context = context_map.get(disease_key, "\n".join(context_map.values()))
            result = qa_pipeline(question=question, context=context)
            st.session_state["last_answer"] = result["answer"]

if "last_answer" in st.session_state:
    st.success("Answer:")
    st.write(st.session_state["last_answer"])

# -------------------------------
# Clear Button (Moved to Bottom)
# -------------------------------

st.markdown("---")
st.markdown("### Reset App")

if st.button("🔄 Clear and Start Over"):
    keys_to_clear = ["predicted_label", "confidence", "last_answer", "uploaded_file"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Session reset! You may now upload a new image or ask a new question.")

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")
st.caption("Powered by PyTorch, HuggingFace Transformers & Streamlit | Built for the 3MTT May Showcase")
