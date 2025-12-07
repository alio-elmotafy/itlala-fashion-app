import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import requests
from PIL import Image
import io
import uuid

# --- PAGE CONFIG ---
st.set_page_config(page_title="Itlala Cloud Edition", page_icon="☁️", layout="wide")

# --- SIDEBAR: SECRETS ---
with st.sidebar:
    st.header("🔑 Cloud API Keys")
    gemini_key = st.text_input("Gemini API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password")
    qdrant_url = st.text_input("Qdrant Cluster URL", placeholder="https://xxx.qdrant.tech")
    qdrant_key = st.text_input("Qdrant API Key", type="password")
    
    st.info("Get keys from Qdrant Cloud & Hugging Face (Settings > Tokens)")

# --- HELPER: EMBEDDING FUNCTION (The "Eyes") ---
# This replaces Marqo's internal model. We use Hugging Face API to turn images/text into vectors.
def get_embedding(text=None, image_file=None):
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/clip-ViT-B-32"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Payload logic: CLIP handles either text OR image
    if text:
        payload = {"inputs": text}
    elif image_file:
        # Read image bytes
        img_bytes = image_file.getvalue() 
        payload = img_bytes # Raw bytes for image
    
    response = requests.post(API_URL, headers=headers, data=payload)
    
    if response.status_code != 200:
        st.error(f"HF Error: {response.text}")
        return None
        
    # HF returns a list of embeddings. We take the first one.
    return response.json()

# --- MAIN APP LOGIC ---
if gemini_key and hf_token and qdrant_url and qdrant_key:
    
    # 1. Setup Clients
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    try:
        q_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        # Create collection if not exists (Vector size 512 for CLIP ViT-B-32)
        if not q_client.collection_exists("itlala_closet"):
            q_client.create_collection(
                collection_name="itlala_closet",
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
    except Exception as e:
        st.error(f"Qdrant Connection Failed: {e}")
        st.stop()

    st.title("☁️ Itlala: Cloud Stylist")
    tab1, tab2 = st.tabs(["📤 Upload", "✨ Recommend"])

    # --- TAB 1: UPLOAD ---
    with tab1:
        st.write("Upload clothes to Qdrant Cloud")
        img_file = st.file_uploader("Choose Image", type=['jpg', 'png'])
        desc = st.text_input("Description (e.g., Blue denim jacket)")
        category = st.selectbox("Category", ["Top", "Bottom", "Shoes", "Formal"])
        
        if st.button("Save to Cloud") and img_file and desc:
            with st.spinner("Generating AI Embeddings..."):
                # 1. Get Vector from Hugging Face
                vector = get_embedding(image_file=img_file)
                
                if vector:
                    # 2. Upload to Qdrant
                    idx = str(uuid.uuid4())
                    q_client.upsert(
                        collection_name="itlala_closet",
                        points=[
                            PointStruct(
                                id=idx,
                                vector=vector,
                                payload={"desc": desc, "category": category}
                            )
                        ]
                    )
                    st.success(f"Saved '{desc}' to database!")

    # --- TAB 2: RECOMMENDATION ---
    with tab2:
        event = st.text_input("Event", "Wedding")
        weather = st.text_input("Weather", "Hot")
        
        if st.button("Suggest Outfit"):
            # A. Gemini plans the search
            prompt = f"Suggest 1 distinct clothing item visual description for a {event} in {weather} weather. Return ONLY the description."
            search_query = model.generate_content(prompt).text.strip()
            st.write(f"🕵️ **AI Searching for:** {search_query}")
            
            # B. Convert "Search Query" to Vector (using same HF Model)
            search_vector = get_embedding(text=search_query)
            
            # C. Search Qdrant
            if search_vector:
                hits = q_client.search(
                    collection_name="itlala_closet",
                    query_vector=search_vector,
                    limit=3
                )
                
                if hits:
                    st.subheader("Found in your closet:")
                    for hit in hits:
                        st.info(f"**{hit.payload['desc']}** (Match Score: {hit.score:.2f})")
                else:
                    st.warning("No matching clothes found. Upload more items!")

else:
    st.warning("👈 Please enter all API keys in the sidebar to start.")