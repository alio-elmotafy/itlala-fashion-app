import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import requests
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
    hf_token = st.text_input("HF Token (Write Access)", value=st.secrets.get("HF_TOKEN", ""), type="password")
    qdrant_url = st.text_input("Qdrant URL", value=st.secrets.get("QDRANT_URL", ""))
    qdrant_key = st.text_input("Qdrant Key", value=st.secrets.get("QDRANT_KEY", ""), type="password")

    st.divider()
    
    # Diagnostic Button
    if st.button("🛠️ Check Connectivity"):
        with st.status("Running Diagnostics...", expanded=True):
            # 1. Check Gemini (Listing available models to avoid 404)
            if gemini_key:
                try:
                    genai.configure(api_key=gemini_key)
                    st.write("Checking Gemini Models...")
                    # Get list of supported models for this key
                    all_models = list(genai.list_models())
                    supported = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
                    if supported:
                        st.success(f"✅ Connected! Found: {supported[0]}")
                        st.write(f"Available: {supported}")
                    else:
                        st.error("❌ Connected but no text models found for this key.")
                except Exception as e:
                    st.error(f"❌ Gemini Error: {e}")
            
            # 2. Check Hugging Face (NEW ROUTER URL)
            if hf_token:
                try:
                    # UPDATED URL BASED ON ERROR 410
                    API_URL = "https://router.huggingface.co/models/sentence-transformers/clip-ViT-B-32"
                    headers = {"Authorization": f"Bearer {hf_token}"}
                    response = requests.post(API_URL, headers=headers, json={"inputs": "test connection"})
                    
                    if response.status_code == 200:
                        st.success("✅ Hugging Face (Router) Connected!")
                    else:
                        st.error(f"❌ HF Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"❌ HF Connection Failed: {e}")

# --- HELPER: ROBUST EMBEDDING (NEW URL) ---
def get_embedding(text=None, image_file=None):
    if not hf_token:
        st.error("Missing Hugging Face Token")
        return None
        
    # --- CRITICAL FIX: NEW HF ROUTER URL ---
    API_URL = "https://router.huggingface.co/models/sentence-transformers/clip-ViT-B-32"
    headers = {"Authorization": f"Bearer {hf_token}"}

    try:
        payload = {}
        if text:
            payload = {"inputs": text}
            response = requests.post(API_URL, headers=headers, json=payload)
        elif image_file:
            img_bytes = image_file.getvalue()
            response = requests.post(API_URL, headers=headers, data=img_bytes)
        
        if response.status_code != 200:
            if "loading" in response.text.lower():
                st.warning("Model is loading... waiting 5s.")
                time.sleep(5)
                return get_embedding(text, image_file)
            
            st.error(f"HF API Error: {response.text}")
            return None
            
        return response.json()

    except Exception as e:
        st.error(f"Embedding Network Error: {e}")
        return None

# --- HELPER: SMART GEMINI CALL ---
def get_gemini_response(prompt_text):
    genai.configure(api_key=gemini_key)
    
    # Try names WITHOUT 'models/' prefix to avoid path doubling
    # Also added 'gemini-1.5-flash-latest' which is often safer
    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-pro', 'gemini-1.0-pro']
    
    last_error = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt_text)
            return response.text
        except Exception as e:
            last_error = e
            continue 
            
    st.error(f"Gemini Failed. Error details: {last_error}")
    return None

# --- MAIN APP ---
if gemini_key and hf_token and qdrant_url and qdrant_key:
    
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
                        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                            vector = vector[0]
                            
                        if isinstance(vector, list) and isinstance(vector[0], float):
                            try:
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
                        else:
                            st.error(f"Invalid Vector: {str(vector)[:50]}...")

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
                        
                        if isinstance(search_vec, list) and isinstance(search_vec[0], float):
                            hits = q_client.search(
                                collection_name="itlala_closet",
                                query_vector=search_vec,
                                limit=3
                            )
                            
                            if hits:
                                st.subheader("Best matches from your closet:")
                                for hit in hits:
                                    st.success(f"**{hit.payload['desc']}** (Match: {int(hit.score*100)}%)")
                            else:
                                st.warning("No close matches found.")
                        else:
                            st.error("Could not vectorize search query.")

else:
    st.info("👈 Please enter your API keys in the sidebar to start.")
