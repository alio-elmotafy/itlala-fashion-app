import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from PIL import Image
import uuid

# --- PAGE CONFIG ---
st.set_page_config(page_title="Itlala: Gemini Edition", page_icon="💎", layout="wide")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Using Gemini for Vision & Embeddings (No Hugging Face needed)")
    
    # We only need ONE key now!
    gemini_key = st.text_input("Gemini API Key", value=st.secrets.get("GEMINI_KEY", ""), type="password")
    qdrant_url = st.text_input("Qdrant URL", value=st.secrets.get("QDRANT_URL", ""))
    qdrant_key = st.text_input("Qdrant Key", value=st.secrets.get("QDRANT_KEY", ""), type="password")

    if st.button("🔌 Connect"):
        if gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                st.success("Gemini Connected!")
            except Exception as e:
                st.error(f"Error: {e}")

# --- HELPER: GEMINI EMBEDDING ---
def get_gemini_embedding(text):
    # This replaces Hugging Face. We use Google's specialized embedding model.
    try:
        # 'models/text-embedding-004' is the latest standard from Google
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
    # Using your powerful model for vision
    vision_model = genai.GenerativeModel('models/gemini-2.0-flash-exp') 
    
    # 2. Setup Qdrant
    try:
        q_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    except Exception as e:
        st.error(f"Qdrant Connection Error: {e}")
        st.stop()

    st.title("💎 Itlala: Pure Gemini Architecture")
    tab1, tab2 = st.tabs(["📤 Upload Clothes", "✨ Get Recommendation"])

    # --- TAB 1: UPLOAD (Image -> Text Description -> Vector) ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            img_file = st.file_uploader("Upload Item", type=['jpg', 'png', 'jpeg'])
            category = st.selectbox("Category", ["Top", "Bottom", "Dress", "Shoes", "Accessory"])
        
        with col2:
            if st.button("Save Item") and img_file:
                with st.spinner("Gemini is analyzing the image..."):
                    try:
                        # A. Open Image
                        image = Image.open(img_file)
                        
                        # B. Ask Gemini to describe it (Vision)
                        prompt = "Describe this fashion item in detail (color, fabric, style, pattern). Be concise."
                        response = vision_model.generate_content([prompt, image])
                        description = response.text
                        st.info(f"**AI Description:** {description}")
                        
                        # C. Convert Description to Vector (Embedding)
                        vector = get_gemini_embedding(description)
                        
                        # D. Save to Qdrant
                        if vector:
                            # Check/Create Collection (Size 768 for Gemini Embeddings)
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
                                            "short_name": description[:50]+"..." # Snippet for display
                                        }
                                    )
                                ]
                            )
                            st.success("✅ Saved to Wardrobe!")
                            
                    except Exception as e:
                        st.error(f"Processing Error: {e}")

    # --- TAB 2: RECOMMENDATION ---
    with tab2:
        event = st.text_input("Event", "Outdoor Wedding")
        weather = st.text_input("Weather", "Sunny and 30 degrees")
        
        if st.button("Suggest Outfit"):
            if not q_client.collection_exists("itlala_gemini"):
                st.warning("Please upload items first!")
            else:
                with st.spinner("Styling..."):
                    # 1. Gemini Plans the Outfit
                    plan_prompt = f"Describe 1 ideal clothing item for a {event} in {weather}. Return ONLY the visual description."
                    search_query = vision_model.generate_content(plan_prompt).text
                    st.write(f"🔍 **Looking for:** {search_query}")
                    
                    # 2. Convert Query to Vector
                    search_vec = get_gemini_embedding(search_query)
                    
                    # 3. Search Qdrant
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

else:
    st.info("👈 Enter Gemini & Qdrant Keys to start.")
