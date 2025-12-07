import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from huggingface_hub import InferenceClient
from PIL import Image
import uuid
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Itlala Cloud Edition", page_icon="☁️", layout="wide")

# --- SIDEBAR: SECRETS ---
with st.sidebar:
    st.header("🔑 Cloud API Keys")
    
    # Try to load from secrets first, otherwise use text input
    default_gemini = st.secrets.get("GEMINI_KEY", "")
    default_hf = st.secrets.get("HF_TOKEN", "")
    default_q_url = st.secrets.get("QDRANT_URL", "")
    default_q_key = st.secrets.get("QDRANT_KEY", "")

    gemini_key = st.text_input("Gemini API Key", value=default_gemini, type="password")
    hf_token = st.text_input("Hugging Face Token", value=default_hf, type="password")
    qdrant_url = st.text_input("Qdrant Cluster URL", value=default_q_url)
    qdrant_key = st.text_input("Qdrant API Key", value=default_q_key, type="password")

# --- HELPER: EMBEDDING FUNCTION (Fixed for New HF API) ---
def get_embedding(text=None, image_file=None):
    # We use the official client to avoid URL errors
    client = InferenceClient(token=hf_token)
    
    # Using a standard, reliable CLIP model
    model_id = "openai/clip-vit-base-patch32"

    try:
        if text:
            # For text, CLIP expects inputs in a specific way or we use a feature extraction call
            # To simplify, we will use the feature_extraction task
            response = client.feature_extraction(text, model=model_id)
            return response
            
        elif image_file:
            # Open image properly
            image = Image.open(image_file)
            # Send image to HF
            response = client.feature_extraction(image, model=model_id)
            return response
            
    except Exception as e:
        st.error(f"Hugging Face Error: {e}")
        return None

# --- MAIN APP LOGIC ---
if gemini_key and hf_token and qdrant_url and qdrant_key:
    
    # 1. Setup Gemini (Added Error Handling)
    try:
        genai.configure(api_key=gemini_key)
        # Fallback to 'gemini-pro' if 1.5-flash gives NotFound
        model = genai.GenerativeModel('gemini-1.5-flash') 
    except Exception as e:
        st.error(f"Gemini Config Error: {e}")
        st.stop()
    
    # 2. Setup Qdrant
    try:
        q_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
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
        category = st.selectbox("Category", ["Top", "Bottom", "Shoes", "Formal", "Dress"])
        
        if st.button("Save to Cloud") and img_file and desc:
            with st.spinner("Generating AI Embeddings..."):
                vector = get_embedding(image_file=img_file)
                
                if vector:
                    # Check if vector is a list (sometimes HF returns list of lists)
                    if isinstance(vector[0], list):
                        vector = vector[0]

                    idx = str(uuid.uuid4())
                    try:
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
                    except Exception as e:
                        st.error(f"Qdrant Save Error: {e}")

    # --- TAB 2: RECOMMENDATION ---
    with tab2:
        event = st.text_input("Event", "Wedding")
        weather = st.text_input("Weather", "Hot")
        
        if st.button("Suggest Outfit"):
            # A. Gemini plans the search
            try:
                prompt = f"Suggest 1 distinct clothing item visual description for a {event} in {weather} weather. Return ONLY the description, nothing else."
                response = model.generate_content(prompt)
                search_query = response.text.strip()
                st.write(f"🕵️ **AI Searching for:** {search_query}")
                
                # B. Convert to Vector
                search_vector = get_embedding(text=search_query)
                
                if search_vector:
                    if isinstance(search_vector[0], list):
                        search_vector = search_vector[0]

                    # C. Search Qdrant
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
            
            except Exception as e:
                st.error(f"AI Process Error: {e}")
                # Fallback diagnosis
                if "404" in str(e) or "NotFound" in str(e):
                    st.warning("Tip: Check if your Gemini API Key is active and supports gemini-1.5-flash.")

else:
    st.warning("👈 Please enter all API keys in the sidebar (or Secrets) to start.")
