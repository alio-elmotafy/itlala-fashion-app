import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from huggingface_hub import InferenceClient
from PIL import Image
import uuid
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Itlala Cloud Edition", page_icon="☁️", layout="wide")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Load secrets or use text inputs
    gemini_key = st.text_input("Gemini API Key", value=st.secrets.get("GEMINI_KEY", ""), type="password")
    hf_token = st.text_input("HF Token (Read/Write)", value=st.secrets.get("HF_TOKEN", ""), type="password")
    qdrant_url = st.text_input("Qdrant URL", value=st.secrets.get("QDRANT_URL", ""))
    qdrant_key = st.text_input("Qdrant Key", value=st.secrets.get("QDRANT_KEY", ""), type="password")

    st.divider()
    # Diagnostic Button
    if st.button("🛠️ Check Connectivity"):
        with st.status("Running Diagnostics...", expanded=True):
            # 1. Check Gemini
            if gemini_key:
                try:
                    genai.configure(api_key=gemini_key)
                    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    st.success(f"Gemini Connected! Available models: {len(models)}")
                    st.write(models) # Show user exactly what names to use
                except Exception as e:
                    st.error(f"Gemini Error: {e}")
            
            # 2. Check Hugging Face
            if hf_token:
                try:
                    client = InferenceClient(token=hf_token)
                    st.write("Testing HF connection...")
                    # Simple ping test
                    client.feature_extraction("hello", model="sentence-transformers/clip-ViT-B-32")
                    st.success("Hugging Face Connected!")
                except Exception as e:
                    st.error(f"HF Error: {e}")

# --- HELPER: ROBUST EMBEDDING ---
def get_embedding(text=None, image_file=None):
    if not hf_token:
        st.error("Missing Hugging Face Token")
        return None
        
    client = InferenceClient(token=hf_token)
    # Using a reliable, public model specifically for CLIP embeddings
    model_id = "sentence-transformers/clip-ViT-B-32"

    try:
        if text:
            return client.feature_extraction(text, model=model_id)
        elif image_file:
            image = Image.open(image_file)
            return client.feature_extraction(image, model=model_id)
    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None

# --- HELPER: SMART GEMINI CALL ---
def get_gemini_response(prompt_text):
    genai.configure(api_key=gemini_key)
    
    # List of models to try in order of preference
    # 'gemini-1.5-flash' is fast, 'gemini-pro' is the reliable backup
    models_to_try = ['models/gemini-1.5-flash', 'models/gemini-1.5-flash-001', 'models/gemini-pro']
    
    last_error = None
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_text)
            return response.text
        except Exception as e:
            last_error = e
            continue # Try next model
            
    # If all fail
    st.error(f"All Gemini models failed. Last error: {last_error}")
    return None

# --- MAIN APP ---
if gemini_key and hf_token and qdrant_url and qdrant_key:
    
    # Initialize Qdrant
    try:
        q_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    except Exception as e:
        st.error(f"Qdrant Connection Error: {e}")
        st.stop()

    st.title("☁️ Itlala: Cloud Stylist")
    tab1, tab2 = st.tabs(["📤 Upload Clothes", "✨ Get Recommendation"])

    # --- TAB 1: UPLOAD ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            img_file = st.file_uploader("Upload Item", type=['jpg', 'png', 'jpeg'])
            desc = st.text_input("Description", placeholder="e.g., Red floral summer dress")
            category = st.selectbox("Category", ["Top", "Bottom", "Dress", "Shoes", "Accessory"])
        
        with col2:
            if st.button("Save Item") and img_file and desc:
                with st.spinner("Processing..."):
                    vector = get_embedding(image_file=img_file)
                    
                    if vector is not None:
                        # Handle varied API return types (list vs nested list)
                        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                            vector = vector[0]
                            
                        try:
                            # Create collection if needed (Safe check)
                            if not q_client.collection_exists("itlala_closet"):
                                q_client.create_collection(
                                    collection_name="itlala_closet",
                                    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
                                )
                            
                            q_client.upsert(
                                collection_name="itlala_closet",
                                points=[
                                    PointStruct(
                                        id=str(uuid.uuid4()),
                                        vector=vector,
                                        payload={"desc": desc, "category": category}
                                    )
                                ]
                            )
                            st.success(f"✅ Saved: {desc}")
                        except Exception as e:
                            st.error(f"Database Error: {e}")

    # --- TAB 2: RECOMMENDATION ---
    with tab2:
        event = st.text_input("Event", "Outdoor Wedding")
        weather = st.text_input("Weather", "Sunny and 30 degrees")
        
        if st.button("Suggest Outfit"):
            if not q_client.collection_exists("itlala_closet"):
                st.warning("Please upload some clothes in Tab 1 first!")
            else:
                with st.spinner("Thinking..."):
                    # 1. Gemini Plan
                    plan_prompt = f"Describe 1 ideal clothing item for a {event} in {weather}. Return ONLY the visual description."
                    search_query = get_gemini_response(plan_prompt)
                    
                    if search_query:
                        st.info(f"🤖 AI is looking for: **{search_query}**")
                        
                        # 2. Vector Search
                        search_vec = get_embedding(text=search_query)
                        if isinstance(search_vec, list) and len(search_vec) > 0 and isinstance(search_vec[0], list):
                            search_vec = search_vec[0]
                            
                        hits = q_client.search(
                            collection_name="itlala_closet",
                            query_vector=search_vec,
                            limit=3
                        )
                        
                        if hits:
                            st.subheader("Best matches from your closet:")
                            for hit in hits:
                                score = hit.score
                                st.success(f"**{hit.payload['desc']}** (Match: {int(score*100)}%)")
                        else:
                            st.warning("No close matches found.")

else:
    st.info("👈 Please enter your API keys in the sidebar to start.")
