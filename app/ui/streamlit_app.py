"""Streamlit entrypoint. User-facing copy is Turkish; keep code and comments in English."""
import streamlit as st
import sys
import os
from dotenv import load_dotenv

# LangChain loaders and splitters for direct ingestion
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.core.rag_chain import app
# KRİTİK DOKUNUŞ: Kilidi aşmak için zaten açık olan veritabanı bağlantısını çağırıyoruz
from app.core.nodes import vectorstore

st.set_page_config(page_title="AskMyDocs Yapay Zeka", page_icon="🤖", layout="wide")

st.title("🤖 AskMyDocs: Gelişmiş RAG Asistanı")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.container():
    col1, col2 = st.columns([1, 10])
    with col1:
        with st.popover("➕", help="Döküman ekle"):
            uploaded_file = st.file_uploader("Dosya seç", type=['pdf', 'txt'], label_visibility="collapsed")
            if uploaded_file is not None:
                # 1. Dosyayı fiziksel olarak kaydet
                save_path = os.path.join(BASE_DIR, "data", uploaded_file.name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 2. Otomatik Ingest İşlemi
                with st.status(f"🚀 {uploaded_file.name} işleniyor...", expanded=True) as status:
                    try:
                        st.write("Döküman okunuyor...")
                        if uploaded_file.name.endswith(".pdf"):
                            loader = PyPDFLoader(save_path)
                        else:
                            loader = TextLoader(save_path)
                        docs = loader.load()

                        st.write("Metinler parçalanıyor (Chunking)...")
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        splits = text_splitter.split_documents(docs)

                        st.write("Vektörler Qdrant veritabanına yazılıyor...")
                        vectorstore.add_documents(splits)

                        status.update(label="✅ Bilgi Bankası Güncellendi!", state="complete", expanded=False)
                        st.success(f"Kaydedildi ve {len(splits)} parça sisteme işlendi: {uploaded_file.name}")
                    except Exception as e:
                        status.update(label="❌ Hata Oluştu!", state="error")
                        st.error(f"İşlem sırasında hata: {e}")

if prompt := st.chat_input("Dökümanlarınız hakkında soru sorun..."):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.status("🛠️ Çalışıyor...", expanded=True) as status:
            st.write("🔍 Yönlendirme (router) çalışıyor...")

            user_turn = sum(1 for m in st.session_state.messages if m["role"] == "user")
            inputs = {
                "question": prompt,
                "is_conversation_opening": user_turn == 1,
            }
            final_output = app.invoke(inputs)

            if final_output.get("chat_mode"):
                st.write("💬 Sohbet modu (döküman ve internet kullanılmadı).")
            elif final_output.get("web_search"):
                st.write("🌐 İnternet araması yapıldı.")
            else:
                st.write("📄 Dökümanlar tarandı.")

            if not final_output.get("chat_mode"):
                st.write("⚖️ Kalite kontrolü yapıldı.")
            st.write("✍️ Yanıt oluşturuluyor...")

            status.update(label="✅ Tamamlandı", state="complete", expanded=False)

        final_answer = final_output.get("generation", "Üzgünüm, yanıt üretilemedi.")
        st.markdown(final_answer)

        if not final_output.get("chat_mode") and final_output.get("search_count", 0) > 0:
            st.info(f"💡 Bu yanıt için {final_output['search_count']} web kaynağı kullanıldı.")

    st.session_state.messages.append({"role": "assistant", "content": final_answer})