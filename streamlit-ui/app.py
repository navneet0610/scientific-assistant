import streamlit as st
import requests
import os
import sys
from utils import translate_categories

# Dynamically add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import settings
from core.settings import BASE_DIR

STATIC_IMG_PATH = os.path.join(BASE_DIR, "multimodal_rag/static/images")

st.set_page_config(page_title="🤖 Voila!", layout="wide")

# Force light theme using custom CSS
st.markdown("""
    <style>
    body {
        background-color: #faf6ff;
        color: #2c1e3e;
    }
    .stTextInput > div > input {
        color: #2c1e3e;
        background-color: white;
    }
    .css-1cpxqw2 {  /* Hacky way to overwrite dark theme background */
        background-color: #faf6ff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Branding
with st.sidebar:
    st.markdown("""
           <h1 style="
               font-size: 28px;
               font-weight: 800;
               color: #5f2b8a;
               text-shadow: 1px 1px 2px #d3bdf0;
               margin-bottom: 0.2rem;
           ">
               <a href="http://localhost:8501" target="_self" style="text-decoration: none; color: inherit;">🤖 Voila!</a>
           </h1>
       """, unsafe_allow_html=True)

    st.markdown("""
        <div style="
            font-size: 16px;
            font-weight: 500;
            color: #4b3c5d;
            margin-bottom: 1.5rem;
            text-shadow: 0.5px 0.5px 1px #dcd3e7;
        ">
            Your Scientific Research Assistant
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            font-size: 0.95rem;
            color: #3a2d4e;
        ">
            💡 <strong>Crafted with Intelligence</strong><br>
            Trained over <strong>400K+ research papers & 45K+ images</strong> 🧠<br><br>
            ⚠️ Disclaimer: <em>Primarily tuned for Computer Science literature — so Voila! shines brightest in tech & AI domains 🤓📚</em>
        </div>
    """, unsafe_allow_html=True)

# --- Header
st.title("🔍 Ready to dive into your next scientific adventure?")
st.markdown("Let’s explore, discover, and decode knowledge — one paper at a time! ✨📚")

# --- Input Section
with st.form("search_form", clear_on_submit=False):
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("Search", placeholder="Enter your query 🙋")
    with col2:
        image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    with col3:
        go = st.form_submit_button("Go 🚀")

# --- Loading & API Call
results = None
if go:
    if not query and not image:
        st.warning("Please provide either a query or an image.")
    else:
        with st.spinner("🧐 Searching..."):
            files = {"image": image} if image else {}
            data = {"query": query, "is_image": str(bool(image)).lower()}
            try:
                response = requests.post("http://localhost:8000/multimodal/search/", data=data, files=files)
                results = response.json()
            except Exception as e:
                st.error(f"Failed to fetch results: {e}")

# --- Display Results
filtered_results = []

# Filter out results where score > 41
if results:
    filtered_results = [item for item in results if item['score'] <= 41]

# If no results are left after filtering, display a message
if not filtered_results:
    st.markdown("### Uh-oh! 😦 Sorry, we don't have papers on your query in our data.")
else:
    for idx, item in enumerate(filtered_results):
        score = item['score']

        st.markdown("---")
        colL, colR = st.columns([3, 2])

        with colL:
            st.markdown(f"### {item.get('title', '')}")
            if item.get("abstract"):
                st.markdown(f"**Abstract:** {item['abstract']}")
            if item.get("caption"):
                st.markdown(f"**Image Description:** {item['caption']}")
            if item.get("citationCount"):
                st.markdown(f"**Citations:** {item['citationCount']}")
            if item.get("authors"):
                st.markdown(f"**Authors:** {item['authors']}")
            raw_categories = item.get("categories", "")
            if isinstance(raw_categories, str):
                raw_categories = raw_categories.strip().split()  # Split by whitespace
            translated_categories = translate_categories(raw_categories)
            if translated_categories:
                st.markdown(f"**Categories:** {', '.join(translated_categories)}")
            if item.get("journal"):
                st.markdown(f"**Journal:** {item['journal']}")
            if item.get("license"):
                st.markdown(f"**License:** [{item['license']}]({item['license']})")

            if item.get("arxiv_id"):
                st.markdown(f"[Download PDF](https://arxiv.org/pdf/{item['arxiv_id']}.pdf)")

        with colR:
            if item.get("images"):
                images = item["images"]
                initial_images = images[:4]  # First 4 images
                remaining_images = images[4:]  # Remaining images

                # Display the first 4 images in a 2x2 grid
                for i in range(0, len(initial_images), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(initial_images):
                            img = initial_images[i + j]
                            full_image_path = os.path.join(STATIC_IMG_PATH, img["image_name"])
                            try:
                                cols[j].image(full_image_path, caption=img.get("caption", ""), use_container_width=True)
                            except Exception as e:
                                cols[j].warning(f"Could not load image: {img['image_name']} — {e}")

                # Show more / Show less section
                if remaining_images:
                    with st.expander("Show more images"):
                        for i in range(0, len(remaining_images), 4):
                            row = st.columns(4)
                            for j in range(4):
                                if i + j < len(remaining_images):
                                    img = remaining_images[i + j]
                                    full_image_path = os.path.join(STATIC_IMG_PATH, img["image_name"])
                                    try:
                                        row[j].image(full_image_path, caption=img.get("caption", ""),
                                                     use_container_width=True)
                                    except Exception as e:
                                        row[j].warning(f"Could not load image: {img['image_name']} — {e}")

        # Displaying the rounded tile based on score
        with colR:
            if score < 16:
                st.markdown(
                    '<div style="position: absolute; top: 10px; right: 10px; background-color: green; color: white; padding: 5px 10px; border-radius: 20px; font-weight: bold;">Voila! Preferred</div>',
                    unsafe_allow_html=True)
            elif 16 <= score < 30:
                st.markdown(
                    '<div style="position: absolute; top: 10px; right: 10px; background-color: yellow; color: #2c1e3e; padding: 5px 10px; border-radius: 20px; font-weight: bold;">Moderate Similarity</div>',
                    unsafe_allow_html=True)
            elif 30 <= score < 41:
                st.markdown(
                    '<div style="position: absolute; top: 10px; right: 10px; background-color: red; color: white; padding: 5px 10px; border-radius: 20px; font-weight: bold;">Low Similarity</div>',
                    unsafe_allow_html=True)
