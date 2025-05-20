import streamlit as st 
st.set_page_config(page_title="AI-Powered LeafScan Assistant", layout="centered")

import torch import torch.nn.functional as F import torchvision.transforms as transforms from PIL import Image import gdown import os from transformers import pipeline import time

-------------------------------

Custom CSS for background images

-------------------------------

st.markdown( f""" <style> .stApp {{ background-image: url("https://raw.githubusercontent.com/Ayo-tech-ai/DeeptechShowcase/refs/heads/main/main_background.jpeg"); background-size: cover; background-attachment: fixed; background-repeat: no-repeat; background-position: center; }} section[data-testid="stSidebar"] > div:first-child {{ background-image: url("https://raw.githubusercontent.com/Ayo-tech-ai/DeeptechShowcase/refs/heads/main/sidebar_background.jpeg"); background-size: cover; background-repeat: no-repeat; background-position: center; }} .stApp, .css-1d391kg, .css-1v0mbdj {{ background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent overlay for contrast */ }} </style> """, unsafe_allow_html=True )

-------------------------------

UI Title & Description

-------------------------------

st.title("🌾 AI-Powered LeafScan Assistant") st.markdown("An AI assistant for detecting rice leaf diseases and providing expert guidance")

-------------------------------

Sidebar (Optional Info)

-------------------------------

with st.sidebar: st.header("About") st.write("This app uses a Convolutional Neural Network (CNN) to classify rice leaf diseases and a transformer-based NLP model to answer user questions.") st.markdown("Built for: 3MTT Monthly Showcase") st.markdown("Creator: Ayoola Mujib Ayodele") st.markdown("GitHub Repo")

-------------------------------

1. Load CNN Model from Google Drive

-------------------------------

MODEL_PATH = "RiceClassifier.pth" FILE_ID = "13nlieOIczZPmbCaA8M2AlefOrXINTXyL"

@st.cache_resource def load_cnn_model(): if not os.path.exists(MODEL_PATH): url = f"https://drive.google.com/uc?id={FILE_ID}" gdown.download(url, MODEL_PATH, quiet=False) model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False) model.eval() return model

cnn_model = load_cnn_model()

-------------------------------

2. Class Names

-------------------------------

class_names = [ 'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bakanae', 'brown_spot', 'grassy_stunt_virus', 'healthy_rice_plant', 'narrow_brown_spot', 'ragged_stunt_virus', 'rice_blast', 'rice_false_smut', 'sheath_blight', 'sheath_rot', 'stem_rot', 'tungro_virus' ]

-------------------------------

3. NLP Q&A Model

-------------------------------

@st.cache_resource def load_qa_model(): return pipeline("question-answering", model="deepset/roberta-base-squad2")

qa_pipeline = load_qa_model()

-------------------------------

4. Disease-Specific Knowledge Base

-------------------------------

context_map = { "rice_blast": "Rice Blast is caused by Magnaporthe oryzae. It can lead to rotten neck symptoms and severe yield loss. It can be managed using resistant varieties or fungicides like tricyclazole.",

"bacterial_leaf_blight": "Bacterial Leaf Blight is caused by Xanthomonas oryzae. It can cause leaf wilting and up to 70% yield loss. It can be managed with resistant varieties or copper-based sprays.",

"bacterial_leaf_streak": "Bacterial Leaf Streak is caused by Xanthomonas oryzicola. It can reduce photosynthesis and grain filling. It can be managed using resistant varieties or copper hydroxide treatments.",

"bakanae": "Bakanae disease is caused by Fusarium fujikuroi. It can lead to tall, weak seedlings and plant death. It can be managed by hot water seed treatment or carbendazim application.",

"brown_spot": "Brown Spot is caused by Bipolaris oryzae. It can reduce yield by damaging older leaves. It can be managed with mancozeb or potassium-rich fertilizers.",

"grassy_stunt_virus": "Grassy Stunt is caused by a virus spread by brown planthoppers. It can lead to stunted plants with few grains. It can be managed by planting resistant varieties or controlling vectors.",

"narrow_brown_spot": "Narrow Brown Spot is caused by Cercospora janseana. It can reduce grain quality and photosynthesis. It can be managed using potassium fertilizer or propiconazole spray.",

"ragged_stunt_virus": "Ragged Stunt Virus is transmitted by brown planthoppers. It can cause twisted leaves and poor grain development. It can be managed using resistant varieties or IPM strategies.",

"rice_false_smut": "False Smut is caused by Ustilaginoidea virens. It can contaminate rice grains with spore balls and toxins. It can be managed by spraying propiconazole or using clean seeds.",

"sheath_blight": "Sheath Blight is caused by Rhizoctonia solani. It can cause lodging and 20–50% yield loss. It can be managed with validamycin or wider plant spacing.",

"sheath_rot": "Sheath Rot is caused by Sarocladium oryzae. It can lead to unfilled grains and rot. It can be managed using carbendazim or Trichoderma-based treatments.",

"stem_rot": "Stem Rot is caused by Sclerotium oryzae. It can result in lodging and poor grain filling. It can be managed by crop rotation or silicon application.",

"tungro_virus": "Tungro is caused by two viruses spread by green leafhoppers. It can cause stunted, yellow-orange plants. It can be managed using resistant cultivars or neem sprays.",

"healthy_rice_plant": "This plant appears to be healthy. No treatment is required. You may ask about disease prevention or general crop care if you wish."

}

-------------------------------

5. Image Preprocessing

-------------------------------

def preprocess_image(image): transform = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) ]) image = image.convert("RGB") return transform(image).unsqueeze(0)

-------------------------------

Step 1: Upload + Diagnose

-------------------------------

st.markdown("## Step 1: Upload & Diagnose")

uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"], key="uploaded_file")

if uploaded_file: image = Image.open(uploaded_file) st.image(image, caption="Uploaded Image", use_container_width=True)

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

Show CNN result

if "predicted_label" in st.session_state: st.success(f"Disease Predicted: {st.session_state.predicted_label}") st.info(f"Confidence Level: {st.session_state.confidence:.2f}%")

-------------------------------

Step 2: Ask the Assistant

-------------------------------

st.markdown("## Step 2: Ask the Smart Assistant")

with st.form(key="qa_form"): question = st.text_input("Ask a question about rice diseases (e.g. How do I treat rice blast?)") submit = st.form_submit_button("Submit")

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

if "last_answer" in st.session_state: st.success("Answer:") st.write(st.session_state["last_answer"])

-------------------------------

Clear Button (Moved to Bottom)

-------------------------------

st.markdown("---") st.markdown("### Reset App")

if st.button("🔄 Clear and Start Over"): keys_to_clear = ["predicted_label", "confidence", "last_answer", "uploaded_file"] for key in keys_to_clear: if key in st.session_state: del st.session_state[key] st.success("Session reset! You may now upload a new image or ask a new question.")

-------------------------------

Footer

-------------------------------

st.markdown("---") st.caption("Powered by PyTorch, HuggingFace Transformers & Streamlit | Built for the 3MTT May Showcase")

