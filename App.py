import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from PIL import Image
import uuid
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Itlala: Gemini Edition", page_icon="💎", layout="wide")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Using 'Gemini Flash Latest' (Auto-Detect)")
    
    gemini_key = st.text_input("Gemini API Key", value=st.secrets.get("GEMINI_KEY", ""), type="password")
    qdrant_url = st.text_input("Qdrant URL", value=st.secrets.get("QDRANT_URL", ""))
    qdrant_key = st.text_input("Qdrant Key", value=st.secrets.get("QDRANT_KEY", ""), type="password")

    if st.button("🔌 Connect & Test"):
        if gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                # Test the model connection immediately
                model = genai.GenerativeModel('models/gemini-flash-latest')
                response = model.generate_content("Hello")
                st.success("✅ Connected Successfully!")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# --- HELPER: GEMINI EMBEDDING ---
def get_gemini_embedding(text):
    try:
        # Adding delay to respect rate limits
        time.sleep(1) 
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
            title="Fashion Item"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Gemini Embedding Error: {e}")
        return None

# --- MAIN APP ---
if gemini_key and qdrant_url and qdrant_key:
    
    # 1. Setup Gemini
    genai.configure(api_key=gemini_key)
    
    # --- FIX: Using the generic alias found in your diagnostics list ---
    # This automatically picks the stable Flash version for your account
    vision_model = genai.GenerativeModel('models/gemini-flash-latest') 
    
    # 2. Setup Qdrant
    try:
        q_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    except Exception as e:
        st.error(f"Qdrant Connection Error: {e}")
        st.stop()

    st.title("💎 Itlala: Gemini Architecture")
    tab1, tab2 = st.tabs(["📤 Upload Clothes", "✨ Get Recommendation"])

    # --- TAB 1: UPLOAD ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            img_file = st.file_uploader("Upload Item", type=['jpg', 'png', 'jpeg'])
            category = st.selectbox("Category", ["Top", "Bottom", "Dress", "Shoes", "Accessory"])
        
        with col2:
            if st.button("Save Item") and img_file:
                with st.spinner("Analyzing image..."):
                    try:
                        image = Image.open(img_file)
                        
                        prompt = "Describe this fashion item in detail (color, fabric, style). Be concise."
                        response = vision_model.generate_content([prompt, image])
                        description = response.text
                        st.info(f"**Description:** {description}")
                        
                        vector = get_gemini_embedding(description)
                        
                        if vector:
                            if not q_client.collection_exists("itlala_gemini"):
                                q_client.create_collection(
                                    collection_name="itlala_gemini",
                                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                                )
                            
                            q_client.upsert(
                                collection_name="itlala_gemini",
                                points=[
                                    PointStruct(
                                        id=str(uuid.uuid4()),
                                        vector=vector,
                                        payload={
                                            "desc": description, 
                                            "category": category,
                                            "short_name": description[:50]+"..."
                                        }
                                    )
                                ]
                            )
                            st.success("✅ Saved to Wardrobe!")
                            
                    except Exception as e:
                        # Detailed error handling to help debug
                        if "404" in str(e):
                             st.error("Model Error: 'gemini-flash-latest' not found. Your key might be restricted.")
                        elif "429" in str(e):
                             st.error("Quota Error: Too many requests. Please wait a moment.")
                        else:
                            st.error(f"Error: {e}")

    # --- TAB 2: RECOMMENDATION ---
    with tab2:
        event = st.text_input("Event", "Outdoor Wedding")
        weather = st.text_input("Weather", "Sunny")
        
        if st.button("Suggest Outfit"):
            if not q_client.collection_exists("itlala_gemini"):
                st.warning("Please upload items first!")
            else:
                with st.spinner("Styling..."):
                    try:
                        plan_prompt = f"Describe 1 ideal clothing item for a {event} in {weather}. Return ONLY the visual description."
                        search_query = vision_model.generate_content(plan_prompt).text
                        st.write(f"🔍 **Looking for:** {search_query}")
                        
                        search_vec = get_gemini_embedding(search_query)
                        
                        if search_vec:
                            hits = q_client.search(
                                collection_name="itlala_gemini",
                                query_vector=search_vec,
                                limit=3
                            )
                            
                            if hits:
                                st.subheader("Found in Closet:")
                                for hit in hits:
                                    st.success(f"**Item:** {hit.payload['short_name']}\n\nMatch Score: {int(hit.score*100)}%")
                            else:
                                st.warning("No matches found.")
                    except Exception as e:
                        st.error(f"Styling Error: {e}")

else:
    st.info("👈 Enter Keys to start.")
