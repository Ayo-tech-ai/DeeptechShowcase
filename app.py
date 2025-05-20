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
    "rice_blast": """Rice Blast is caused by the fungus Magnaporthe oryzae. It typically forms diamond-shaped or spindle lesions with gray centers and brown margins. The disease can lead to rotten neck symptoms, severely reducing grain production and causing total crop failure in severe cases. It spreads through airborne spores and thrives in high humidity. Management includes using resistant varieties like IR64 and applying fungicides such as tricyclazole.""",

    "bacterial_leaf_blight": """Bacterial Leaf Blight is caused by the bacterium Xanthomonas oryzae pv. oryzae. Early symptoms include water-soaked leaf tips that turn yellow and wilt. It can lead to up to 70% yield loss and spreads rapidly through rain, irrigation water, and infected seeds. Farmers can manage it using resistant rice varieties, copper-based sprays, clean seeds, and crop rotation.""",

    "bacterial_leaf_streak": """Bacterial Leaf Streak is caused by Xanthomonas oryzae pv. oryzicola, a bacterial pathogen. It appears as narrow, water-soaked streaks between the veins that later turn yellow-brown. The disease reduces photosynthesis and grain filling, causing yield losses up to 30%. Management involves resistant varieties like IR50 and spraying with copper hydroxide.""",

    "bakanae": """Bakanae disease is caused by the fungus Fusarium fujikuroi. It results in tall, thin, pale seedlings with weak stems that may die early. The infection spreads through contaminated seeds and soil, often leading to significant plant loss. Control measures include hot water seed treatment and fungicide application, particularly with carbendazim.""",

    "brown_spot": """Brown Spot is caused by the fungus Bipolaris oryzae. It produces brown circular lesions with a light center and dark margin, especially on older leaves. The disease can reduce yield by up to 90% under poor soil conditions. Management includes improving soil fertility with potassium or silicon and applying fungicides like mancozeb.""",

    "grassy_stunt_virus": """Rice Grassy Stunt Virus (RGSV) is a viral disease spread by the brown planthopper. Infected plants show pale, narrow leaves and stunted growth with few or no grains. It can result in complete crop failure. Control includes planting resistant varieties such as IR36 and practicing synchronized planting and insect management.""",

    "narrow_brown_spot": """Narrow Brown Spot is caused by the fungus Cercospora janseana. It causes elongated, brown streaks mostly along leaf veins. The disease affects photosynthesis and reduces grain quality. It is managed by applying potassium fertilizer and fungicides like propiconazole.""",

    "ragged_stunt_virus": """Rice Ragged Stunt Virus (RRSV) is transmitted by brown planthoppers. It causes ragged, twisted, and shortened leaves with stunted plants and malformed grains. The virus can reduce yields by over 50%. Management involves using resistant rice varieties like IR72 and controlling vector populations through integrated pest management.""",

    "rice_false_smut": """Rice False Smut is caused by the fungus Ustilaginoidea virens. It replaces rice grains with greenish-black spore balls and contaminates the harvest with mycotoxins. The disease spreads during flowering in humid conditions. Management includes spraying propiconazole at the booting stage and using clean seeds.""",

    "sheath_blight": """Sheath Blight is caused by the soil-borne fungus Rhizoctonia solani. Symptoms include oval lesions on the sheath near the waterline that spread upward, leading to lodging. It can cause 20–50% yield loss under dense planting. Management includes wider spacing, using resistant varieties, and applying fungicides like validamycin.""",

    "sheath_rot": """Sheath Rot is caused by the fungus Sarocladium oryzae. It produces brown to dark lesions on the sheath, leading to rotting of spikelets and unfilled grains. The disease spreads via air and wounds, and may lead to yield losses of up to 85%. Control involves spraying carbendazim and applying biological control agents like Trichoderma.""",

    "stem_rot": """Stem Rot is caused by the fungus Sclerotium oryzae, identified by the presence of black sclerotia inside stems. It causes stem lodging, poor grain filling, and can reduce yield significantly. The pathogen survives in the soil and crop residue. Control includes crop rotation, silicon application, and deep plowing after harvest.""",

    "tungro_virus": """Rice Tungro is a viral disease caused by a combination of Rice Tungro Bacilliform Virus (RTBV) and Rice Tungro Spherical Virus (RTSV), spread by green leafhoppers. Infected plants show yellow-orange discoloration and stunted growth. It can cause total crop failure in susceptible varieties. Control includes planting resistant cultivars like IR36 and using neem-based insecticides to manage vector populations.""",

    "healthy_rice_plant": """This plant appears to be healthy. No treatment is required. You may ask about disease prevention or general crop care if you wish."""
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
