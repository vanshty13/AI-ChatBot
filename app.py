import streamlit as st
import requests
import time
import json
import datetime
import PyPDF2
import io
import base64
import os
from typing import Optional, Dict, Any
from deep_translator import GoogleTranslator

# Set your backend server URL
BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="QuantAI Real Estate Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "response_detail" not in st.session_state:
    st.session_state.response_detail = "detailed"
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_typing" not in st.session_state:
    st.session_state.is_typing = False
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = {}
if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False
if "active_file_id" not in st.session_state:
    st.session_state.active_file_id = None
if "file_metadata" not in st.session_state:
    st.session_state.file_metadata = {}  # Store file metadata: {file_id: {filename, file_type, upload_time}}
# MODIFIED: Initialize language code in session state
if "language_code" not in st.session_state:
    st.session_state.language_code = "en"

def translate_text(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """Translate text to target language using deep-translator"""
    try:
        if target_lang == "en":
            return text  # No translation needed for English
        
        # Map language codes for deep-translator
        lang_mapping = {
            "es": "es",  # Spanish
            "fr": "fr",  # French
            "de": "de",  # German
            "zh": "zh",  # Chinese
            "ja": "ja"   # Japanese
        }
        
        target_lang_code = lang_mapping.get(target_lang, "en")
        if target_lang_code == "en":
            return text
            
        # Use Google Translator from deep-translator
        translator = GoogleTranslator(source=source_lang, target=target_lang_code)
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def translate_response(response: str, target_lang: str) -> str:
    """Translate AI response to user's selected language"""
    if target_lang == "en":
        return response
    return translate_text(response, target_lang, "en")

def translate_user_message(message: str, target_lang: str) -> str:
    """Translate user message to English for AI processing"""
    if target_lang == "en":
        return message
    return translate_text(message, "en", target_lang)

def get_image_base64(image_path):
    """Convert image to base64 data URI for embedding in HTML."""
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
            return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""

def create_header():
    img_base64 = get_image_base64("logo/quantai__logo.jpg")
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 0.5rem;">
            <img src="{img_base64}" width="50" style="vertical-align: middle;"/>
            &nbsp;
            <span style="font-size: 3rem; font-weight: 700; 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                -webkit-background-clip: text; 
                -webkit-text-fill-color: transparent; 
                display: inline-block;">
                QuantAI Real Estate
            </span>
        </div>
        <div style="text-align: center;">
            <h2 style="font-size: 1.5rem; color: #666; margin-top: 0;">
                Your AI-Powered Property Assistant
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Stats cards section
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Properties", "1,000+", "📊")
    with col2:
        st.metric("Regions", "6", "🌍")
    with col3:
        st.metric("Heritage", "150+", "🏛")
    with col4:
        st.metric("Success Rate", "98%", "✅")

def create_sidebar():
    with st.sidebar:
        st.markdown("### 🤖 Jim - AI Assistant")
        
        # Mode indicator
        lang_code = st.session_state.get("language_code", "en")
        sidebar_mode_messages = {
            "en": {
                "rag": "🤖 *RAG Mode* - File-specific answers",
                "general": "🤖 *General Mode* - General knowledge"
            },
            "es": {
                "rag": "🤖 *Modo RAG* - Respuestas específicas del archivo",
                "general": "🤖 *Modo General* - Conocimiento general"
            },
            "fr": {
                "rag": "🤖 *Mode RAG* - Réponses spécifiques au fichier",
                "general": "🤖 *Mode Général* - Connaissances générales"
            },
            "de": {
                "rag": "🤖 *RAG-Modus* - Dateispezifische Antworten",
                "general": "🤖 *Allgemeiner Modus* - Allgemeines Wissen"
            },
            "zh": {
                "rag": "🤖 *RAG模式* - 基于文件的特定回答",
                "general": "🤖 *通用模式* - 通用知识"
            },
            "ja": {
                "rag": "🤖 *RAGモード* - ファイル固有の回答",
                "general": "🤖 *一般モード* - 一般知識"
            }
        }
        
        messages = sidebar_mode_messages.get(lang_code, sidebar_mode_messages["en"])
        if st.session_state.get("active_file_id"):
            st.success(messages["rag"])
        else:
            st.info(messages["general"])
        
        st.markdown("---")
        
        # Settings panel
        with st.expander("⚙ Settings", expanded=False):
            st.markdown("#### 🎨 Appearance")
            
            # Theme selector
            theme = st.selectbox(
                "Theme",
                ["Light", "Dark"],
                index=1 if st.session_state.theme == "dark" else 0,
                key="theme_selector"
            )
            
            if theme.lower() != st.session_state.theme:
                st.session_state.theme = theme.lower()
                st.success(f"✅ Theme changed to {theme}")
            
            # Response detail
            detail = st.selectbox(
                "Response Detail",
                ["Brief", "Detailed"],
                index=1 if st.session_state.response_detail == "detailed" else 0,
                key="detail_selector"
            )
            
            if detail.lower() != st.session_state.response_detail:
                st.session_state.response_detail = detail.lower()
                st.success(f"✅ Response detail set to {detail}")
            
            st.markdown("---")
            st.markdown("#### 🌍 Language")
            
            languages = {
                "English": "en",
                "Spanish": "es", 
                "French": "fr",
                "German": "de",
                "Chinese": "zh",
                "Japanese": "ja"
            }
            
            selected_lang = st.selectbox(
                "Select Language",
                list(languages.keys()),
                index=list(languages.keys()).index([k for k, v in languages.items() if v == st.session_state.language_code][0]), # MODIFIED: Set index based on current session state
                key="language_selector"
            )
            
            # MODIFIED: Store the selected language code immediately
            st.session_state.language_code = languages[selected_lang]
            
            # Send language preference to backend
            success, error = post_language(languages[selected_lang])
            if success:
                st.success(f"✅ Language set to {selected_lang}")
            else:
                st.error(f"❌ Error setting language: {error}")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("#### 🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑 Clear Chat", use_container_width=True, key="clear_chat_btn"):
                st.session_state.chat_history = []
                st.success("✅ Chat cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Memory", use_container_width=True, key="reset_memory_btn"):
                success, error = reset_conversation()
                if success:
                    st.session_state.chat_history = []
                    st.success("✅ Memory reset!")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {error}")
        
        if st.button("📋 Start Qualification", use_container_width=True, key="qualification_btn"):
            response, error = start_buyer_qualification()
            if response:
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": datetime.datetime.now().strftime("%H:%M")
                })
                st.success("✅ Qualification started!")
                st.rerun()
            else:
                st.error(f"❌ Error: {error}")
        
        st.markdown("---")
        
        # File upload
        lang_code = st.session_state.get("language_code", "en")
        upload_messages = {
            "en": {
                "title": "#### 📄 Upload Documents",
                "info": "Upload files to enable RAG mode for file-specific answers"
            },
            "es": {
                "title": "#### 📄 Subir Documentos",
                "info": "Sube archivos para habilitar el modo RAG para respuestas específicas del archivo"
            },
            "fr": {
                "title": "#### 📄 Télécharger des Documents",
                "info": "Téléchargez des fichiers pour activer le mode RAG pour des réponses spécifiques aux fichiers"
            },
            "de": {
                "title": "#### 📄 Dokumente Hochladen",
                "info": "Laden Sie Dateien hoch, um den RAG-Modus für dateispezifische Antworten zu aktivieren"
            },
            "zh": {
                "title": "#### 📄 上传文档",
                "info": "上传文件以启用RAG模式进行文件特定回答"
            },
            "ja": {
                "title": "#### 📄 ドキュメントをアップロード",
                "info": "ファイルをアップロードしてRAGモードを有効にし、ファイル固有の回答を取得"
            }
        }
        
        messages = upload_messages.get(lang_code, upload_messages["en"])
        st.markdown(messages["title"])
        st.info(messages["info"])
        
        uploader_text = {
            "en": "Upload documents (CSV, TXT, PDF)",
            "es": "Subir documentos (CSV, TXT, PDF)",
            "fr": "Télécharger des documents (CSV, TXT, PDF)",
            "de": "Dokumente hochladen (CSV, TXT, PDF)",
            "zh": "上传文档 (CSV, TXT, PDF)",
            "ja": "ドキュメントをアップロード (CSV, TXT, PDF)"
        }.get(lang_code, "Upload documents (CSV, TXT, PDF)")
        
        uploaded_files = st.file_uploader(
            uploader_text,
            type=['csv', 'txt', 'pdf'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        # Process uploaded files
        for uploaded_file in uploaded_files:
            # Check if this file is already processed
            file_already_processed = False
            for existing_file in st.session_state.uploaded_files:
                if existing_file.get('filename') == uploaded_file.name:
                    file_already_processed = True
                    break
            
            if not file_already_processed:
                file_content = process_uploaded_file(uploaded_file)
                if uploaded_file.name.endswith('.csv'):
                    file_type = 'csv'
                elif uploaded_file.name.endswith('.txt'):
                    file_type = 'txt'
                elif uploaded_file.name.endswith('.pdf'):
                    file_type = 'pdf'
                else:
                    file_type = 'txt'
                
                success, error, file_id = upload_file_to_backend(uploaded_file.name, file_content, file_type)
                if success:
                    st.success(f"✅ {uploaded_file.name} - RAG knowledge base created!")
                    # Store file metadata in session state
                    st.session_state.file_metadata[file_id] = {
                        'filename': uploaded_file.name,
                        'file_type': file_type,
                        'upload_time': datetime.datetime.now().isoformat()
                    }
                else:
                    st.error(f"❌ {uploaded_file.name} - {error}")
        
        # Fetch file list on first load if not present
        if not st.session_state.uploaded_files:
            file_list = get_file_list()
            st.session_state.uploaded_files = file_list["files"]
            st.session_state.active_file_id = file_list.get("active_file_id")
        
        # File selection dropdown
        if st.session_state.uploaded_files:
            file_names = ["General Mode"] + [f["filename"] for f in st.session_state.uploaded_files]
            file_ids = [None] + [f["file_id"] for f in st.session_state.uploaded_files]
            idx = 0
            if st.session_state.active_file_id:
                try:
                    idx = file_ids.index(st.session_state.active_file_id)
                except Exception:
                    idx = 0
            
            select_text = {
                "en": "Select Active File",
                "es": "Seleccionar Archivo Activo",
                "fr": "Sélectionner le Fichier Actif",
                "de": "Aktive Datei Auswählen",
                "zh": "选择活动文件",
                "ja": "アクティブファイルを選択"
            }.get(lang_code, "Select Active File")
            
            selected = st.selectbox(select_text, file_names, index=idx, key="active_file_select")
            selected_file_id = file_ids[file_names.index(selected)]
            
            # Update active file ID and clear it if General Mode is selected
            if selected == "General Mode":
                st.session_state.active_file_id = None
                # Clear active file on backend
                clear_active_file()
                general_mode_text = {
                    "en": "General Mode: No file selected. Chat will use general knowledge.",
                    "es": "Modo General: No se seleccionó archivo. El chat usará conocimiento general.",
                    "fr": "Mode Général: Aucun fichier sélectionné. Le chat utilisera les connaissances générales.",
                    "de": "Allgemeiner Modus: Keine Datei ausgewählt. Chat verwendet allgemeines Wissen.",
                    "zh": "通用模式：未选择文件。聊天将使用通用知识。",
                    "ja": "一般モード：ファイルが選択されていません。チャットは一般知識を使用します。"
                }.get(lang_code, "General Mode: No file selected. Chat will use general knowledge.")
                st.info(general_mode_text)
            else:
                st.session_state.active_file_id = selected_file_id
                if selected_file_id:
                    if set_active_file(selected_file_id):
                        st.success(f"Active file set: {selected}")
                    else:
                        st.error("Failed to set active file.")
                    st.info(f"Active file: {selected}")
        else:
            no_files_text = {
                "en": "No files uploaded. Upload CSV or TXT files to create a knowledge base.",
                "es": "No se subieron archivos. Sube archivos CSV o TXT para crear una base de conocimientos.",
                "fr": "Aucun fichier téléchargé. Téléchargez des fichiers CSV ou TXT pour créer une base de connaissances.",
                "de": "Keine Dateien hochgeladen. Laden Sie CSV- oder TXT-Dateien hoch, um eine Wissensbasis zu erstellen.",
                "zh": "未上传文件。上传CSV或TXT文件以创建知识库。",
                "ja": "ファイルがアップロードされていません。CSVまたはTXTファイルをアップロードして知識ベースを作成してください。"
            }.get(lang_code, "No files uploaded. Upload CSV or TXT files to create a knowledge base.")
            st.info(no_files_text)

def apply_theme_settings():
    """Apply theme settings to the app"""
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117 !important;
            color: white !important;
        }
        .stSidebar {
            background-color: #262730 !important;
        }
        .stSidebar .stMarkdown {
            color: white !important;
        }
        .stChatMessage {
            background-color: #262730 !important;
            color: white !important;
        }
        .stChatMessage .stMarkdown {
            color: white !important;
        }
        .stSelectbox > div > div {
            background-color: #262730 !important;
            color: white !important;
        }
        .stButton > button {
            background-color: #262730 !important;
            color: white !important;
            border: 1px solid #444 !important;
        }
        .stButton > button:hover {
            background-color: #444 !important;
            border-color: #666 !important;
        }
        .stTextInput > div > div > input {
            background-color: #262730 !important;
            color: white !important;
            border: 1px solid #444 !important;
        }
        .stFileUploader {
            background-color: #262730 !important;
            color: white !important;
        }
        .stExpander {
            background-color: #262730 !important;
            color: white !important;
        }
        .stInfo {
            background-color: #1e3a8a !important;
            color: white !important;
        }
        .stSuccess {
            background-color: #166534 !important;
            color: white !important;
        }
        .stError {
            background-color: #dc2626 !important;
            color: white !important;
        }
        .stMetric {
            background-color: #262730 !important;
            color: white !important;
        }
        .stSpinner {
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff !important;
            color: #262730 !important;
        }
        .stSidebar {
            background-color: #f8fafc !important;
        }
        .stSidebar .stMarkdown {
            color: #262730 !important;
        }
        .stChatMessage {
            background-color: #f8fafc !important;
            color: #262730 !important;
            border: 1px solid #e2e8f0 !important;
        }
        .stChatMessage .stMarkdown {
            color: #262730 !important;
        }
        .stSelectbox > div > div {
            background-color: white !important;
            color: #262730 !important;
            border: 1px solid #d1d5db !important;
        }
        .stButton > button {
            background-color: white !important;
            color: #262730 !important;
            border: 1px solid #d1d5db !important;
        }
        .stButton > button:hover {
            background-color: #f9fafb !important;
            border-color: #9ca3af !important;
        }
        .stTextInput > div > div > input {
            background-color: white !important;
            color: #262730 !important;
            border: 1px solid #d1d5db !important;
        }
        .stFileUploader {
            background-color: white !important;
            color: #262730 !important;
            border: 1px solid #d1d5db !important;
        }
        .stExpander {
            background-color: white !important;
            color: #262730 !important;
            border: 1px solid #d1d5db !important;
        }
        .stInfo {
            background-color: #dbeafe !important;
            color: #1e40af !important;
            border: 1px solid #93c5fd !important;
        }
        .stSuccess {
            background-color: #dcfce7 !important;
            color: #166534 !important;
            border: 1px solid #86efac !important;
        }
        .stError {
            background-color: #fef2f2 !important;
            color: #dc2626 !important;
            border: 1px solid #fca5a5 !important;
        }
        .stMetric {
            background-color: white !important;
            color: #262730 !important;
            border: 1px solid #e5e7eb !important;
        }
        .stSpinner {
            color: #4f46e5 !important;
        }
        /* Fix for chat input in light mode */
        .stChatInput > div {
            background-color: white !important;
            border: 1px solid #d1d5db !important;
        }
        .stChatInput input {
            background-color: white !important;
            color: #262730 !important;
        }
        /* Fix for sidebar elements in light mode */
        .stSidebar .stSelectbox label {
            color: #262730 !important;
        }
        .stSidebar .stMarkdown h1, 
        .stSidebar .stMarkdown h2, 
        .stSidebar .stMarkdown h3, 
        .stSidebar .stMarkdown h4 {
            color: #262730 !important;
        }
        /* Fix for main content headers in light mode */
        .stMarkdown h1, 
        .stMarkdown h2, 
        .stMarkdown h3, 
        .stMarkdown h4 {
            color: #262730 !important;
        }
        /* Fix for captions in light mode */
        .stChatMessage .stCaption {
            color: #6b7280 !important;
        }
        </style>
        """, unsafe_allow_html=True)

def create_chat_interface():
    """Create the main chat interface"""
    lang_code = st.session_state.get("language_code", "en")
    
    chat_title = {
        "en": "### 💬 Chat with Jim",
        "es": "### 💬 Chat con Jim",
        "fr": "### 💬 Chat avec Jim",
        "de": "### 💬 Chat mit Jim",
        "zh": "### 💬 与Jim聊天",
        "ja": "### 💬 Jimとのチャット"
    }.get(lang_code, "### 💬 Chat with Jim")
    
    st.markdown(chat_title)
    
    # Show current mode and active file
    if st.session_state.active_file_id:
        # Find the active file name
        active_file_name = "Unknown"
        for file_info in st.session_state.uploaded_files:
            if file_info.get('file_id') == st.session_state.active_file_id:
                active_file_name = file_info.get('filename', 'Unknown')
                break
        
        rag_mode_text = {
            "en": f"🤖 *RAG Mode*: Using file '{active_file_name}' for responses",
            "es": f"🤖 *Modo RAG*: Usando archivo '{active_file_name}' para respuestas",
            "fr": f"🤖 *Mode RAG*: Utilisation du fichier '{active_file_name}' pour les réponses",
            "de": f"🤖 *RAG-Modus*: Verwende Datei '{active_file_name}' für Antworten",
            "zh": f"🤖 *RAG模式*: 使用文件 '{active_file_name}' 进行回答",
            "ja": f"🤖 *RAGモード*: ファイル '{active_file_name}' を使用して回答"
        }.get(lang_code, f"🤖 *RAG Mode*: Using file '{active_file_name}' for responses")
        
        st.success(rag_mode_text)
        st.info(f"📄 Active File ID: {st.session_state.active_file_id}")
    else:
        general_mode_text = {
            "en": "📄 *General Mode*: Chat with Jim about real estate topics. Upload a file to switch to RAG mode.",
            "es": "📄 *Modo General*: Chat con Jim sobre temas inmobiliarios. Sube un archivo para cambiar al modo RAG.",
            "fr": "📄 *Mode Général*: Chat avec Jim sur les sujets immobiliers. Téléchargez un fichier pour passer au mode RAG.",
            "de": "📄 *Allgemeiner Modus*: Chat mit Jim über Immobilienthemen. Laden Sie eine Datei hoch, um zum RAG-Modus zu wechseln.",
            "zh": "📄 *通用模式*: 与Jim聊天关于房地产话题。上传文件以切换到RAG模式。",
            "ja": "📄 *一般モード*: Jimと不動産トピックについてチャット。ファイルをアップロードしてRAGモードに切り替え。"
        }.get(lang_code, "📄 *General Mode*: Chat with Jim about real estate topics. Upload a file to switch to RAG mode.")
        
        st.info(general_mode_text)
    
    # Display chat history
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(entry["content"])
                if entry.get("timestamp"):
                    st.caption(f"🕐 {entry['timestamp']}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(entry["content"])
                if entry.get("timestamp"):
                    st.caption(f"🕐 {entry['timestamp']}")
    
    # Chat input with dynamic placeholder based on language
    lang_code = st.session_state.get("language_code", "en")
    placeholder_text = {
        "en": "Type your message here...",
        "es": "Escribe tu mensaje aquí...",
        "fr": "Tapez votre message ici...",
        "de": "Geben Sie Ihre Nachricht hier ein...",
        "zh": "在此输入您的消息...",
        "ja": "ここにメッセージを入力してください..."
    }.get(lang_code, "Type your message here...")
    
    if prompt := st.chat_input(placeholder_text, key="main_chat_input"):
        # Add user message to chat history
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.datetime.now().strftime("%H:%M")
        }
        st.session_state.chat_history.append(user_message)
        
        # Display user message immediately
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
            st.caption(f"🕐 {user_message['timestamp']}")
        
        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            spinner_text = {
                "en": "Jim is thinking...",
                "es": "Jim está pensando...",
                "fr": "Jim réfléchit...",
                "de": "Jim denkt nach...",
                "zh": "Jim正在思考...",
                "ja": "Jimが考えています..."
            }.get(lang_code, "Jim is thinking...")
            
            with st.spinner(spinner_text):
                # MODIFIED: Get language code from session state
                lang_code = st.session_state.get("language_code", "en")
                
                # Translate user message to English for AI processing if not English
                translated_prompt = translate_user_message(prompt, lang_code)
                
                # Modify prompt based on response detail setting
                if st.session_state.response_detail == "brief":
                    enhanced_prompt = f"{translated_prompt} (Please provide a brief, concise response)"
                else:
                    enhanced_prompt = f"{translated_prompt} (Please provide a detailed, comprehensive response)"
                
                # MODIFIED: Pass language code to the chat function
                response, error = post_chat_message(enhanced_prompt, lang_code)
                
                if error:
                    st.error(f"❌ Error: {error}")
                    # Add error to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Error: {error}",
                        "timestamp": datetime.datetime.now().strftime("%H:%M")
                    })
                elif response:
                    # Translate AI response to user's selected language
                    translated_response = translate_response(response, lang_code)
                    st.write(translated_response)
                    timestamp = datetime.datetime.now().strftime("%H:%M")
                    st.caption(f"🕐 {timestamp}")
                    
                    # Add translated response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": translated_response,
                        "timestamp": timestamp
                    })
                else:
                    error_msg = "No response received from the server."
                    st.error(f"❌ {error_msg}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}",
                        "timestamp": datetime.datetime.now().strftime("%H:%M")
                    })
        
        # Force rerun to update the chat display
        st.rerun()

def create_suggestions():
    """Create quick suggestion buttons"""
    lang_code = st.session_state.get("language_code", "en")
    
    # Dynamic suggestions based on language
    suggestions_data = {
        "en": {
            "title": "### 💡 Quick Suggestions",
            "subtitle": "Click any suggestion to get started:",
            "suggestions": [
                "🏠 Find colonial properties",
                "📊 Get market analysis", 
                "💰 Investment advice",
                "🏛 Property valuation",
                "🌍 Heritage listings",
                "📍 Regional insights"
            ]
        },
        "es": {
            "title": "### 💡 Sugerencias Rápidas",
            "subtitle": "Haz clic en cualquier sugerencia para comenzar:",
            "suggestions": [
                "🏠 Encontrar propiedades coloniales",
                "📊 Obtener análisis de mercado", 
                "💰 Consejos de inversión",
                "🏛 Valoración de propiedades",
                "🌍 Listados de patrimonio",
                "📍 Información regional"
            ]
        },
        "fr": {
            "title": "### 💡 Suggestions Rapides",
            "subtitle": "Cliquez sur une suggestion pour commencer:",
            "suggestions": [
                "🏠 Trouver des propriétés coloniales",
                "📊 Obtenir une analyse de marché", 
                "💰 Conseils d'investissement",
                "🏛 Évaluation de propriété",
                "🌍 Listes du patrimoine",
                "📍 Informations régionales"
            ]
        },
        "de": {
            "title": "### 💡 Schnelle Vorschläge",
            "subtitle": "Klicken Sie auf einen Vorschlag, um zu beginnen:",
            "suggestions": [
                "🏠 Koloniale Immobilien finden",
                "📊 Marktanalyse erhalten", 
                "💰 Anlageberatung",
                "🏛 Immobilienbewertung",
                "🌍 Denkmallisten",
                "📍 Regionale Einblicke"
            ]
        },
        "zh": {
            "title": "### 💡 快速建议",
            "subtitle": "点击任何建议开始:",
            "suggestions": [
                "🏠 查找殖民时期房产",
                "📊 获取市场分析", 
                "💰 投资建议",
                "🏛 房产估值",
                "🌍 遗产清单",
                "📍 区域洞察"
            ]
        },
        "ja": {
            "title": "### 💡 クイック提案",
            "subtitle": "提案をクリックして開始:",
            "suggestions": [
                "🏠 コロニアル物件を探す",
                "📊 市場分析を取得", 
                "💰 投資アドバイス",
                "🏛 物件評価",
                "🌍 遺産リスト",
                "📍 地域情報"
            ]
        }
    }
    
    data = suggestions_data.get(lang_code, suggestions_data["en"])
    st.markdown(data["title"])
    st.markdown(data["subtitle"])
    
    suggestions = data["suggestions"]
    
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        col = cols[i % 3]
        if col.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            # Trigger the suggestion
            handle_suggestion(suggestion)

def handle_suggestion(suggestion):
    """Handle when a suggestion is clicked"""
    # Add user message to chat history
    user_message = {
        "role": "user",
        "content": suggestion,
        "timestamp": datetime.datetime.now().strftime("%H:%M")
    }
    st.session_state.chat_history.append(user_message)
    
    # MODIFIED: Get language code from session state
    lang_code = st.session_state.get("language_code", "en")
    
    # Translate suggestion to English for AI processing if not English
    translated_suggestion = translate_user_message(suggestion, lang_code)
    
    # MODIFIED: Pass language code to the chat function
    response, error = post_chat_message(translated_suggestion, lang_code)
    
    if error:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"❌ Error: {error}",
            "timestamp": datetime.datetime.now().strftime("%H:%M")
        })
    elif response:
        # Translate AI response to user's selected language
        translated_response = translate_response(response, lang_code)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": translated_response,
            "timestamp": datetime.datetime.now().strftime("%H:%M")
        })
    
    # Force rerun to update display
    st.rerun()

# API Functions
# MODIFIED: Add language_code parameter
def post_chat_message(message: str, language_code: str) -> tuple[Optional[str], Optional[str]]:
    """Send POST to /chat with user message and return response text or error."""
    try:
        print(f"DEBUG: Sending message to {BASE_URL}/chat: {message}")
        # Prepare request payload with session, file, and language context
        payload = {
            "message": message,
            "session_id": "default_session",
            "language": language_code  # MODIFIED: Add language to payload
        }
        # Add active file context if available and not in General Mode
        print(f"DEBUG: Current active_file_id: {st.session_state.get('active_file_id', 'None')}")
        if (
            hasattr(st.session_state, 'active_file_id') and 
            st.session_state.active_file_id is not None and 
            st.session_state.active_file_id != "None" and
            st.session_state.active_file_id != None
        ):
            print(f"DEBUG: Including active file ID: {st.session_state.active_file_id}")
            payload["active_file_id"] = st.session_state.active_file_id
        else:
            print("DEBUG: General Mode - No active file included in payload.")
            # Don't include active_file_id in payload for General Mode
        resp = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        print(f"DEBUG: Response status: {resp.status_code}")
        if resp.status_code == 200:
            response_data = resp.json()
            print(f"DEBUG: Response data: {response_data}")
            response_text = response_data.get("response", "")
            if response_text:
                return response_text, None
            else:
                return None, "Empty response from server"
        else:
            try:
                error_data = resp.json()
                error_msg = error_data.get("error", f"HTTP {resp.status_code}")
            except:
                error_msg = f"HTTP {resp.status_code}"
            return None, error_msg
    except requests.exceptions.Timeout:
        return None, "Request timeout. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Please check if the backend server is running."
    except Exception as e:
        print(f"DEBUG: Exception in post_chat_message: {e}")
        return None, f"Connection error: {e}"

def post_language(language_code: str) -> tuple[bool, Optional[str]]:
    """Send POST to /language with selected language code."""
    try:
        resp = requests.post(
            f"{BASE_URL}/language", 
            json={"language": language_code}, 
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        if resp.status_code == 200 and resp.json().get("success"):
            return True, None
        else:
            return False, resp.json().get("error", "Unknown error")
    except Exception as e:
        return False, f"Connection error: {e}"

def start_buyer_qualification() -> tuple[Optional[str], Optional[str]]:
    """Start the buyer qualification process."""
    try:
        resp = requests.post(f"{BASE_URL}/buyer-qualification", timeout=30)
        if resp.status_code == 200:
            return resp.json().get("response", "[No response]"), None
        else:
            return None, resp.json().get("error", "Unknown error")
    except Exception as e:
        return None, f"Connection error: {e}"

def reset_conversation() -> tuple[bool, Optional[str]]:
    """Reset the conversation memory."""
    try:
        resp = requests.post(f"{BASE_URL}/conversation/reset", timeout=10)
        if resp.status_code == 200:
            return True, None
        else:
            return False, resp.json().get("error", "Unknown error")
    except Exception as e:
        return False, f"Connection error: {e}"

def get_user_preferences() -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Get current user preferences."""
    try:
        resp = requests.get(f"{BASE_URL}/user-preferences", timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, resp.json().get("error", "Unknown error")
    except Exception as e:
        return None, f"Connection error: {e}"

def process_uploaded_file(uploaded_file) -> str:
    """Process uploaded file and extract text content."""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        elif uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
            return str(uploaded_file.read(), "utf-8")
        else:
            return f"Unsupported file type: {uploaded_file.type}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

def get_file_list(session_id: str = "default_session"):
    try:
        resp = requests.get(f"{BASE_URL}/file/list/{session_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"files": [], "active_file_id": None}
    except Exception as e:
        return {"files": [], "active_file_id": None}

def set_active_file(file_id: str, session_id: str = "default_session"):
    try:
        resp = requests.post(f"{BASE_URL}/file/active", json={"session_id": session_id, "file_id": file_id}, timeout=10)
        if resp.status_code == 200 and resp.json().get("success"):
            return True
        return False
    except Exception as e:
        return False

def clear_active_file(session_id: str = "default_session"):
    """Clear the active file for a session"""
    try:
        resp = requests.post(f"{BASE_URL}/file/active", json={"session_id": session_id, "file_id": None}, timeout=10)
        if resp.status_code == 200 and resp.json().get("success"):
            return True
        return False
    except Exception as e:
        return False

def upload_file_to_backend(filename: str, content: str, file_type: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Upload file to backend for RAG processing and refresh file list"""
    try:
        session_id = "default_session"
        payload = {
            "filename": filename,
            "content": content,
            "file_type": file_type,
            "session_id": session_id
        }
        resp = requests.post(
            f"{BASE_URL}/file/upload",
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        # Always refresh file list after upload
        file_list = get_file_list(session_id)
        st.session_state.uploaded_files = file_list["files"]
        st.session_state.active_file_id = file_list.get("active_file_id")
        if resp.status_code == 200:
            response_data = resp.json()
            if response_data.get("success"):
                file_id = response_data.get("file_id")
                return True, None, file_id
            else:
                return False, response_data.get("error", "Unknown error"), None
        else:
            return False, f"HTTP {resp.status_code}: {resp.text}", None
    except Exception as e:
        return False, f"Connection error: {e}", None

def clear_rag_knowledge_base() -> tuple[bool, Optional[str]]:
    """Clear RAG knowledge base"""
    try:
        session_id = "default_session"
        resp = requests.delete(
            f"{BASE_URL}/file/clear/{session_id}",
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        
        if resp.status_code == 200:
            return True, None
        else:
            return False, f"HTTP {resp.status_code}: {resp.text}"
            
    except Exception as e:
        return False, f"Connection error: {e}"

def main():
    """Main application function"""
    # Apply theme settings
    apply_theme_settings()
    
    # Create header
    create_header()
    
    # Create sidebar
    create_sidebar()
    
    # Main content area
    st.markdown("---")
    
    # Layout with columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat interface
        create_chat_interface()
        
        # Show quick suggestions if no chat history
        if not st.session_state.chat_history:
            st.markdown("---")
            if st.session_state.get("active_file_id"):
                create_suggestions()
            else:
                lang_code = st.session_state.get("language_code", "en")
                info_messages = {
                    "en": "💡 Try asking about real estate topics, property search, or market analysis!",
                    "es": "💡 ¡Intenta preguntar sobre temas inmobiliarios, búsqueda de propiedades o análisis de mercado!",
                    "fr": "💡 Essayez de poser des questions sur les sujets immobiliers, la recherche de propriétés ou l'analyse de marché !",
                    "de": "💡 Versuchen Sie, Fragen zu Immobilienthemen, Immobiliensuche oder Marktanalyse zu stellen!",
                    "zh": "💡 尝试询问房地产话题、房产搜索或市场分析！",
                    "ja": "💡 不動産トピック、物件検索、市場分析について質問してみてください！"
                }
                st.info(info_messages.get(lang_code, info_messages["en"]))
        
        # Show mode indicator
        lang_code = st.session_state.get("language_code", "en")
        mode_messages = {
            "en": {
                "rag": "🤖 *RAG Mode*: Answers based on uploaded file",
                "general": "🤖 *General Mode*: Chat with general knowledge"
            },
            "es": {
                "rag": "🤖 *Modo RAG*: Respuestas basadas en archivo subido",
                "general": "🤖 *Modo General*: Chat con conocimiento general"
            },
            "fr": {
                "rag": "🤖 *Mode RAG*: Réponses basées sur le fichier téléchargé",
                "general": "🤖 *Mode Général*: Chat avec connaissances générales"
            },
            "de": {
                "rag": "🤖 *RAG-Modus*: Antworten basierend auf hochgeladener Datei",
                "general": "🤖 *Allgemeiner Modus*: Chat mit allgemeinem Wissen"
            },
            "zh": {
                "rag": "🤖 *RAG模式*: 基于上传文件的回答",
                "general": "🤖 *通用模式*: 使用通用知识聊天"
            },
            "ja": {
                "rag": "🤖 *RAGモード*: アップロードされたファイルに基づく回答",
                "general": "🤖 *一般モード*: 一般知識でのチャット"
            }
        }
        
        messages = mode_messages.get(lang_code, mode_messages["en"])
        if st.session_state.get("active_file_id"):
            st.success(messages["rag"])
        else:
            st.info(messages["general"])
    
    with col2:
        # User preferences panel
        st.markdown("### 📊 Your Preferences")
        
        # Force refresh preferences
        preferences, error = get_user_preferences()
        if preferences and not error:
            st.session_state.user_preferences = preferences
            
            if preferences.get('budget_max'):
                st.info(f"💰 Budget: ${preferences['budget_max']:,}")
            if preferences.get('preferred_regions'):
                st.info(f"📍 Regions: {', '.join(preferences['preferred_regions'])}")
            if preferences.get('bedrooms'):
                st.info(f"🛏 Bedrooms: {preferences['bedrooms']}")
            if preferences.get('must_have_features'):
                st.info(f"✨ Features: {', '.join(preferences['must_have_features'])}")
            if preferences.get('investment_focus'):
                st.info("💼 Investment Focus: Yes")
            
            # Show raw preferences for debugging
            with st.expander("🔍 Debug - Raw Preferences"):
                st.json(preferences)
        elif error:
            st.error(f"❌ Error loading preferences: {error}")
        else:
            st.info("No preferences set yet. Start chatting to build your profile!")
        
        # Uploaded files
        if st.session_state.uploaded_files:
            st.markdown("### 📄 Uploaded Files")
            for file_info in st.session_state.uploaded_files:
                st.text(f"📄 {file_info['filename']}")
            
            # RAG Status
            st.markdown("### 🤖 RAG Knowledge Base")
            if st.session_state.get("active_file_id"):
                st.success("✅ Active - Responses based on uploaded files")
            else:
                st.info("📄 File uploaded but not active. Select a file to enable RAG mode.")
            
            # Clear RAG button
            if st.button("🗑 Clear RAG Knowledge Base", use_container_width=True, key="clear_rag_btn"):
                success, error = clear_rag_knowledge_base()
                if success:
                    # Always fetch the file list from backend after clearing
                    file_list = get_file_list()
                    st.session_state.uploaded_files = file_list["files"]
                    st.session_state.active_file_id = file_list.get("active_file_id")
                    st.session_state.file_metadata = {}
                    st.success("✅ RAG knowledge base cleared!")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {error}")
        else:
            st.markdown("### 🤖 RAG Knowledge Base")
            st.info("No files uploaded.")

if __name__=="__main__":
    main()