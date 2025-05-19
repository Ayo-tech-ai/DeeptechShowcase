import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os
from transformers import pipeline

# -------------------------------
# 1. Load CNN Model
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
# 2. Class Labels
# -------------------------------

class_names = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bakanae',
    'brown_spot', 'grassy_stunt_virus', 'healthy_rice_plant',
    'narrow_brown_spot', 'ragged_stunt_virus', 'rice_blast',
    'rice_false_smut', 'sheath_blight', 'sheath_rot',
    'stem_rot', 'tungro_virus'
]

# -------------------------------
# 3. Load NLP Q&A Model
# -------------------------------

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

qa_pipeline = load_qa_model()

# -------------------------------
# 4. Rice Disease Knowledge Base
# -------------------------------

context = """
Rice is affected by various diseases with significant impacts on yield. Bacterial Leaf Blight is caused by the bacterium Xanthomonas oryzae pv. oryzae. It presents symptoms like water-soaked lesions, yellowing, and wilting, and may lead to up to 70% yield loss. Control involves resistant varieties, copper sprays, and crop rotation.

Bacterial Leaf Streak is caused by Xanthomonas oryzae pv. oryzicola. It causes water-soaked streaks and yield loss. Managed with resistant varieties and copper hydroxide.

Bakanae Disease is caused by Fusarium fujikuroi. It leads to tall, thin, fragile seedlings. Managed by seed treatment and carbendazim.

Brown Spot is caused by Bipolaris oryzae. Appears as brown bullseyes. Managed with potassium-rich fertilizers and mancozeb.

Rice Grassy Stunt Virus spreads via brown planthopper, causing pale leaves and sterility. Managed by using IR36 and synchronized planting.

Narrow Brown Spot is caused by Cercospora janseana. Symptoms include narrow lesions. Treated with propiconazole and potassium fertilizer.

Rice Ragged Stunt Virus causes distorted and ragged leaves. Managed with resistant strains and IPM practices.

Rice Blast is caused by Magnaporthe oryzae. Appears as diamond lesions. Managed with resistant varieties and tricyclazole.

Rice False Smut creates black-green balls replacing grains. Managed during flowering with propiconazole.

Sheath Blight is caused by Rhizoctonia solani. Forms sheath lesions and leads to lodging. Managed with spacing and validamycin.

Sheath Rot causes grain shriveling. Managed by Trichoderma and fungicides.

Stem Rot caused by Sclerotium oryzae leads to black sclerotia in stems. Managed by crop rotation and silicon.

Tungro Virus is a dual virus spread by green leafhoppers. Leads to stunting and yellowing. Controlled with neem and resistant varieties.
"""

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
# 6. App Layout
# -------------------------------

st.title("SmartAgri AI Assistant")

st.markdown("### 1. Upload Rice Leaf Image for Disease Prediction")

uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify Disease"):
        with st.spinner("Analyzing image..."):
            input_tensor = preprocess_image(image)
            outputs = cnn_model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

            label = class_names[predicted_class.item()]
            conf = confidence.item() * 100

            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence Level:** {conf:.2f}%")

# -------------------------------
st.markdown("---")
st.markdown("### 2. Ask a Question About Rice Diseases")

with st.form(key="qa_form"):
    question = st.text_input("Type your question here (e.g., How to treat rice blast?)")
    submit = st.form_submit_button("Get Answer")

    if submit and question:
        with st.spinner("Answering..."):
            result = qa_pipeline(question=question, context=context)
            st.success("Answer:")
            st.write(result["answer"])
