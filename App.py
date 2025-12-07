# --- MAIN APP ---
if gemini_key and qdrant_url and qdrant_key:
    
    # 1. Setup Gemini
    genai.configure(api_key=gemini_key)
    
    # --- التغيير هنا: استبدلنا الموديل التجريبي بالموديل المستقر ---
    # gemini-1.5-flash سريع جداً، مجاني، ويدعم الرؤية (Vision) بكفاءة عالية
    vision_model = genai.GenerativeModel('models/gemini-1.5-flash') 
    
    # باقي الكود كما هو...
    # 2. Setup Qdrant
    try:
        q_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
# ...
