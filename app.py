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
    "rice_blast": """Rice Blast, caused by the fungus Magnaporthe oryzae, forms diamond-shaped lesions and can result in "rotten neck" symptoms. It may lead to total crop failure. Airborne spores and high humidity facilitate its spread. Management includes using resistant varieties and tricyclazole fungicide.""",
    "bacterial_leaf_blight": """Bacterial Leaf Blight is caused by Xanthomonas oryzae pv. oryzae. It presents symptoms like water-soaked lesions, yellowing, and wilting, and may lead to up to 70% yield loss. Control involves resistant varieties, copper sprays, and crop rotation.""",
    "bacterial_leaf_streak": """Bacterial Leaf Streak is caused by Xanthomonas oryzae pv. oryzicola. It causes water-soaked streaks and yield loss. Management includes resistant varieties and copper hydroxide.""",
    "bakanae": """Bakanae Disease, caused by Fusarium fujikuroi, produces tall, fragile seedlings. Managed by seed treatment and carbendazim.""",
    "brown_spot": """Brown Spot is caused by Bipolaris oryzae. Lesions appear as brown bullseyes. Treated with mancozeb and balanced fertilization.""",
    "grassy_stunt_virus": """Rice Grassy Stunt Virus (RGSV) leads to pale leaves and stunted growth. Controlled by using resistant varieties like IR36.""",
    "narrow_brown_spot": """Caused by Cercospora janseana, this disease shows up as narrow streaks. Managed with propiconazole and potassium fertilization.""",
    "ragged_stunt_virus": """Rice Ragged Stunt Virus causes ragged leaves and stunted plants. Managed through IPM and resistant strains.""",
    "rice_false_smut": """False Smut forms black-green spore balls. Reduces yield and spreads during flowering. Treated with propiconazole.""",
    "sheath_blight": """Caused by Rhizoctonia solani. Produces sheath lesions. Controlled using validamycin and crop spacing.""",
    "sheath_rot": """Sheath Rot causes grain shriveling and rot. Managed with carbendazim and Trichoderma.""",
    "stem_rot": """Stem Rot involves black sclerotia in the stem. Caused by Sclerotium oryzae. Managed by crop rotation and silicon treatment.""",
    "tungro_virus": """Tungro is caused by two viruses and spread by green leafhoppers. It results in stunting and yellowing. Controlled with neem spray and resistant varieties.""",
    "healthy_rice_plant": """This plant appears to be healthy. No treatment is required. You may ask about disease prevention if you wish."""
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
# 6. Streamlit Interface
# -------------------------------

st.title("Rice Disease Detection + AI_Smart Assistant")

uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Classify Disease"):
        with st.spinner("Classifying with CNN..."):
            input_tensor = preprocess_image(image)
            outputs = cnn_model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

            predicted_label = class_names[predicted_class.item()]
            confidence_score = confidence.item() * 100

            st.success(f"Prediction: **{predicted_label}**")
            st.info(f"Confidence Level: {confidence_score:.2f}%")

            # NLP Assistant — updated to ensure reliable submit + answer
            disease_key = predicted_label.lower().replace(" ", "_")
            if disease_key in context_map:
                context = context_map[disease_key]
                st.subheader(f"Ask about **{predicted_label}**")

                if "show_answer" not in st.session_state:
                    st.session_state.show_answer = False

                question = st.text_input("What do you want to know?", key="user_question")
                submit = st.button("Submit", key="submit_question")

                if submit:
                    st.session_state.show_answer = True

                if st.session_state.show_answer and question:
                    with st.spinner("Thinking..."):
                        result = qa_pipeline(question=question, context=context)
                        st.success("Answer:")
                        st.write(result["answer"])
                        st.session_state.show_answer = False
            else:
                st.warning("No disease-specific advice available for this prediction.")
