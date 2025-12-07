import streamlit as st
import google.generativeai as genai
# Import Qdrant Client correctly
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
except ImportError:
    st.error("Qdrant Library not found. Please add 'qdrant-client>=1.7.0' to requirements.txt")
    st.stop()

from PIL import Image
import uuid
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Itlala: Final Edition", page_icon="👗", layout="wide")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Using Stable Gemini 1.5 Flash")
    
    gemini_key = st.text_input("Gemini API Key", value=st.secrets.get("GEMINI_KEY", ""), type="password")
    qdrant_url = st.text_input("Qdrant URL", value=st.secrets.get("QDRANT_URL", ""))
    qdrant_key = st.text_input("Qdrant Key", value=st.secrets.get("QDRANT_KEY", ""), type="password")

    if st.button("🔌 Test Connection"):
        if gemini_key and qdrant_url:
            try:
                genai.configure(api_key=gemini_key)
                m = genai.GenerativeModel('models/gemini-1.5-flash-latest')
                m.generate_content("hi")
                st.success("Gemini: OK")
                
                q = QdrantClient(url=qdrant_url, api_key=qdrant_key)
                st.success(f"Qdrant: OK (v{st.secrets.get('qdrant_version', 'Detected')})")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# --- HELPER: GEMINI EMBEDDING ---
def get_gemini_embedding(text):
    try:
        # Rate limit protection
        time.sleep(1) 
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
            title="Fashion Item"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None

# --- MAIN APP ---
if gemini_key and qdrant_url and qdrant_key:
    
    # 1. Setup Gemini (Using the SAFE Latest Alias)
    genai.configure(api_key=gemini_key)
    # 'gemini-1.5-flash-latest' auto-resolves to the best stable version you have access to
    vision_model = genai.GenerativeModel('models/gemini-1.5-flash-latest') 
    
    # 2. Setup Qdrant
    try:
        q_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    except Exception as e:
        st.error(f"Qdrant Init Error: {e}")
        st.stop()

    st.title("👗 Itlala: Smart Wardrobe")
    tab1, tab2 = st.tabs(["📤 Upload Clothes", "✨ Get Recommendation"])

    # --- TAB 1: UPLOAD ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            img_file = st.file_uploader("Upload Item", type=['jpg', 'png', 'jpeg'])
            category = st.selectbox("Category", ["Top", "Bottom", "Dress", "Shoes", "Accessory"])
        
        with col2:
            if st.button("Save Item") and img_file:
                with st.spinner("Analyzing..."):
                    try:
                        image = Image.open(img_file)
                        
                        prompt = "Describe this fashion item in detail (color, fabric, style). Be concise."
                        response = vision_model.generate_content([prompt, image])
                        description = response.text
                        st.info(f"**AI Description:** {description}")
                        
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
                        if "404" in str(e):
                            st.error("Gemini Model Not Found. Try checking your API key permissions.")
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
                        # 1. Plan
                        plan_prompt = f"Describe 1 ideal clothing item for a {event} in {weather}. Return ONLY the visual description."
                        search_query = vision_model.generate_content(plan_prompt).text
                        st.write(f"🔍 **Looking for:** {search_query}")
                        
                        # 2. Embed
                        search_vec = get_gemini_embedding(search_query)
                        
                        # 3. Search (With Check)
                        if search_vec:
                            # Explicit check for the search method to debug your error
                            if not hasattr(q_client, 'search'):
                                st.error("CRITICAL: Your Qdrant Client is outdated or corrupted. Please Reboot App.")
                            else:
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
