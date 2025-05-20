# 🌾 AI-Powered LeafScan Assistant

**An AI-powered web application for diagnosing rice leaf diseases and answering related questions using computer vision and natural language processing.**

[Live App →](https://smartagric4ai.streamlit.app)  
[GitHub Repository →](https://github.com/Ayo-tech-ai/DeeptechShowcase)

---

## 💡 About the Project

A lot of farmers in Nigeria often struggle with the timely identification of crop diseases, particularly in rice production. Delays in diagnosis—due to the distance or unavailability of crop management teams—can lead to severe damage and significant yield loss.

The **AI-Powered LeafScan Assistant** bridges this gap by enabling farmers to:

- Instantly **scan rice leaves using their phone**
- **Identify potential diseases** using a CNN-powered model
- **Ask questions** related to symptoms, treatment, or prevention via a smart NLP assistant
- **Take proactive action** or communicate findings to experts before help arrives

This tool makes disease identification faster, more scalable, and accessible directly from a smartphone or PC browser—empowering farmers to make better decisions, earlier.

---

## 🚀 Features

- **CNN-based Image Classification** for 14 rice leaf diseases + healthy condition
- **NLP Assistant** powered by RoBERTa for context-aware Q&A
- **Confidence Scoring** to help users understand prediction reliability
- **Mobile-friendly UI** with upload and Q&A in one place
- **Runs on Streamlit Cloud** – no installation needed
- Model hosted via Google Drive and loaded dynamically using `gdown`

---

## 🧠 Technologies Used

- **Python**
- **PyTorch** – Image Classification Model
- **HuggingFace Transformers** – NLP Q&A Model (RoBERTa)
- **Streamlit** – UI/UX and deployment
- **Google Drive + gdown** – for remote model loading

---

## 🔗 Demo

> [Click here to try the live app](https://smartagric4ai.streamlit.app)

---

## 👤 Creator


Ayoola Mujib Ayodele
LinkedIn Profile
Email: ayodelemujibayoola@gmail.com


---

🤝 Contributions

This project was built for the 3MTT May Showcase as a demonstration of how AI can power scalable and accessible solutions in agriculture and food security.
Suggestions, feedback, and collaborations are welcome!


---

⭐ Show Support

If you find this project helpful or inspiring:

Star the repository

Watch for updates

Share with your network or community



---

© License

This project is currently not licensed.
Please contact the creator for permission to reuse, modify, or extend this work.

---


## 🛠 How to Run Locally

To run the app locally on your machine:

```bash
# Clone this repository
git clone https://github.com/Ayo-tech-ai/DeeptechShowcase
cd DeeptechShowcase

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

> Note: The model will automatically be downloaded from Google Drive on first run.
