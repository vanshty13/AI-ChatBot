from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import Any, Dict, AsyncGenerator, Optional, List
import asyncio
import json
import os
import requests
import re
import csv
import io
from datetime import datetime
from dotenv import load_dotenv
from requests.exceptions import ChunkedEncodingError
import time
import uuid
from sentence_transformers import SentenceTransformer
import numpy as np
from contextlib import asynccontextmanager

# NEW: Import the static data loader
from data_loader import load_static_knowledge_base

load_dotenv()

# NEW: Lifespan event handler for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global static_kb
    static_kb = load_static_knowledge_base()
    print("INFO: Static knowledge base loaded successfully at startup.")
    yield

app = FastAPI(title="QuantAI Real Estate API", description="AI-powered real estate assistant API", lifespan=lifespan)

# Add CORS middleware for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NEW: Global variable to store the static knowledge base
static_kb = []

# In-memory user language preference (for demo purposes)
user_language_preference = {"language": "en"}

# Track conversation state for each user (for demo purposes)
conversation_state = {"first_message": True}

# Property tracking for consistency
property_tracker = {
    "current_property": None,
    "property_history": [],
    "property_details": {}
}

# Conversation memory and user preferences
conversation_memory = {
    "history": [],
    "user_preferences": {
        'budget_min': None,
        'budget_max': None,
        'preferred_regions': [],
        'preferred_property_types': [],
        'bedrooms': None,
        'must_have_features': [],
        'investment_focus': False,
        'heritage_interest': False
    },
    "buyer_qualification": {
        'active': False,
        'step': 0,
        'answers': {}
    }
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# In-memory RAG knowledge base per session
rag_knowledge_bases = {}

# File upload tracking per session
uploaded_files = {}

# In-memory per-session, per-file knowledge bases
file_knowledge_bases = {}  # {session_id: {file_id: {content, embeddings, ...}}}
active_file = {}  # {session_id: file_id}

# Load embedding model once
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print("WARNING: Could not load 'all-MiniLM-L6-v2'. Using dummy embeddings. Error:", e)
    class DummyModel:
        def encode(self, texts):
            import numpy as np
            if isinstance(texts, str):
                texts = [texts]
            return [np.zeros(384) for _ in texts]
    embedding_model = DummyModel()

def compute_embedding(text):
    return embedding_model.encode([text])[0]

def cosine_similarity(a, b):
    # Ensure inputs are numpy arrays
    a = np.array(a) if isinstance(a, list) else a
    b = np.array(b) if isinstance(b, list) else b
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def chunk_and_embed(content):
    # Chunk by paragraphs (double newlines)
    chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
    embeddings = [compute_embedding(chunk) for chunk in chunks]
    return list(zip(chunks, embeddings))

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    active_file_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str

class LanguageRequest(BaseModel):
    language: str

class LanguageResponse(BaseModel):
    language: str
    success: bool
    error: Optional[str] = None

class StreamChatRequest(BaseModel):
    message: str

class BuyerQualificationRequest(BaseModel):
    message: str

class BuyerQualificationResponse(BaseModel):
    response: str

class ConversationSummary(BaseModel):
    message_count: int
    user_preferences: Dict[str, Any]
    buyer_qualification_active: bool
    last_interaction: Optional[str] = None

class ResetResponse(BaseModel):
    message: str

class PropertyInfo(BaseModel):
    address: str
    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    price: Optional[float] = None
    features: Optional[List[str]] = None
    region: Optional[str] = None

class EnhancedChatRequest(BaseModel):
    message: str
    property_address: Optional[str] = None
    property_context: Optional[Dict[str, Any]] = None

class FileUploadRequest(BaseModel):
    filename: str
    content: str
    file_type: str  # 'csv' or 'txt'
    session_id: str

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None
    file_id: Optional[str] = None
    error: Optional[str] = None

class FileInfo(BaseModel):
    filename: str
    file_type: str
    content_preview: str
    upload_time: str

# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "QuantAI Real Estate API is running", "version": "1.0.0"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# NEW: Modified groq_chat_completion to accept static context
def groq_chat_completion(user_message: str, language: str = "en", is_first_message: bool = False, property_context: Optional[Dict] = None, session_id: Optional[str] = None, static_context: Optional[str] = None) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not set.")
    
    # Validate API key format
    if not GROQ_API_KEY.startswith("gsk_"):
        raise HTTPException(status_code=500, detail="Invalid Groq API key format. Should start with 'gsk_'.")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Get conversation context
    context = get_context_summary()
    
    # Check for RAG knowledge base
    rag_content = ""
    if session_id and session_id in rag_knowledge_bases:
        rag_kb = rag_knowledge_bases[session_id]
        if rag_kb.get('knowledge_base'):
            rag_content = f"""
UPLOADED FILE CONTENT (Use this information to answer questions):
{rag_kb['knowledge_base']}

IMPORTANT: When answering questions, use ONLY the information from the uploaded file above. If the question cannot be answered using this information, clearly state that the information is not available in the uploaded file.
"""
    
    # Build property-specific context
    property_info = ""
    if property_context and property_context.get('address'):
        property_info = f"""
CURRENT PROPERTY CONTEXT:
Address: {property_context.get('address', 'Unknown')}
Type: {property_context.get('property_type', 'Unknown')}
Bedrooms: {property_context.get('bedrooms', 'Unknown')}
Bathrooms: {property_context.get('bathrooms', 'Unknown')}
Price: {property_context.get('price', 'Unknown')}
Features: {', '.join(property_context.get('features', []))}
Region: {property_context.get('region', 'Unknown')}

IMPORTANT: Only provide information about this specific property. If you don't have accurate information about this property, say so clearly. Do not guess or make assumptions about property details.
"""
    
    # NEW: Add the static knowledge base context to the prompt
    static_kb_content = ""
    if static_context:
        static_kb_content = f"""
STATIC KNOWLEDGE BASE CONTEXT:
Use this information to answer questions about specific regions, suburbs, and market trends. Prioritize this information for factual real estate data.
{static_context}
"""
    
    # Always use English for Groq API calls - we'll translate the response to user's language
    system_prompt = f"""You are Jim, a professional Real Estate AI Assistant at QuantAI, specializing in New Zealand's property market. 

PROFESSIONAL GUIDELINES:

1. Be conversational and helpful - engage naturally with users
2. When users ask about specific properties, provide accurate information if available
3. If you don't have information about a property, politely say so and offer to help with general market information
4. For general questions about real estate, market trends, or property types, provide helpful insights
5. If users mention a property address, try to provide relevant information, but don't insist on exact matches
6. Be professional but friendly - build rapport with users
7. Offer to help with property searches, market analysis, or investment advice
8. Keep responses concise but informative

Your expertise includes:
- Colonial & heritage properties (260 heritage-listed in your database)
- Modern residential properties
- Market analysis & investment insights
- Regional market knowledge across 6 regions

You have access to 1,000 properties across New Zealand, with a special focus on colonial architecture.

CONVERSATIONAL APPROACH:
- Be welcoming and helpful from the first interaction
- Ask follow-up questions to better understand user needs
- Provide general market insights when specific property info isn't available
- Suggest related topics or services that might be helpful
- Maintain a professional but friendly tone

Your personality:
- Professional and knowledgeable about New Zealand real estate
- Helpful and conversational
- CONCISE - keep responses short and direct
- ACCURATE - provide verified information when available
- Always respond in English (the response will be translated to the user's language)

IMPORTANT: {'This is the FIRST message. Give a brief introduction as Jim from QuantAI and ask how you can help. Keep it short and to the point.' if is_first_message else 'This is NOT the first message. Respond naturally and conversationally. Focus on helping with property searches, market analysis, or investment advice.'}

{rag_content}
{property_info}
{static_kb_content}
Conversation context: {context}"""
    
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.5,  # Reduced temperature for more consistent responses
        "max_tokens": 1024,
        "stream": False,
        "top_p": 0.9,  # Slightly reduced for more focused responses
        "frequency_penalty": 0.1,  # Slight penalty to reduce repetition
        "presence_penalty": 0.1  # Slight penalty to encourage diverse responses
    }
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            print(f"DEBUG: Sending request to Groq API (attempt {attempt+1})")
            resp = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
            print(f"DEBUG: Response status: {resp.status_code}")
            print(f"DEBUG: Response headers: {dict(resp.headers)}")
            if not resp.ok:
                error_detail = f"HTTP {resp.status_code}"
                try:
                    error_json = resp.json()
                    print(f"DEBUG: Error response: {error_json}")
                    if "error" in error_json:
                        error_detail += f": {error_json['error'].get('message', 'Unknown error')}"
                        if resp.status_code == 401:
                            error_detail = "Invalid API key. Please check your GROQ_API_KEY."
                        elif resp.status_code == 429:
                            error_detail = "Rate limit exceeded. Please try again later."
                        elif resp.status_code == 400:
                            error_detail = f"Invalid request: {error_json['error'].get('message', 'Bad request')}"
                except:
                    error_detail += f": {resp.text[:200]}"
                raise HTTPException(status_code=500, detail=f"Groq API error: {error_detail}")
            result = resp.json()
            print(f"DEBUG: Success response: {json.dumps(result, indent=2)}")
            if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                return result["choices"][0]["message"]["content"]
            else:
                raise HTTPException(status_code=500, detail="Groq API returned unexpected response format.")
        except ChunkedEncodingError as ce:
            print(f"DEBUG: ChunkedEncodingError: {ce}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise HTTPException(status_code=500, detail="Groq API connection was interrupted. Please try again.")
        except requests.exceptions.IncompleteRead as ir:
            print(f"DEBUG: IncompleteRead: {ir}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise HTTPException(status_code=500, detail="Groq API returned an incomplete response. Please try again.")
        except HTTPException:
            raise
        except Exception as e:
            print(f"DEBUG: Exception: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")
    
    # If all retries failed, return a default error message
    raise HTTPException(status_code=500, detail="Failed to get response from Groq API after multiple attempts.")

# Streaming endpoint
async def groq_stream_chat_completion(user_message: str, language: str = "en", is_first_message: bool = False, property_context: Optional[Dict] = None, session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, str], None]:
    try:
        response = await asyncio.to_thread(groq_chat_completion, user_message, language, is_first_message, property_context, session_id)
        # Simulate streaming by splitting response into words
        words = response.split()
        for word in words:
            await asyncio.sleep(0.1)  # Small delay to simulate streaming
            yield {"chunk": word + " "}
    except HTTPException as e:
        yield {"error": str(e.detail)}
    except Exception as e:
        yield {"error": f"Server error: {e}"}

def rag_chat_completion(user_query: str, file_content: str, session_id: str, document_name: str = "the uploaded document") -> str:
    """Generate RAG response using only the specific file content with enhanced context"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not set.")
    
    # Validate API key format
    if not GROQ_API_KEY.startswith("gsk_"):
        raise HTTPException(status_code=500, detail="Invalid Groq API key format. Should start with 'gsk_'.")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Create a focused prompt that only uses the specific file content
    system_prompt = f"""You are a helpful AI assistant. The user has uploaded a document and is asking questions about it. 

IMPORTANT: You must answer questions based ONLY on the following document content. Do not use any external knowledge or general information.

DOCUMENT: {document_name}
SESSION: {session_id}

DOCUMENT CONTENT:
{file_content}

INSTRUCTIONS:
1. Answer the user's question using ONLY the information from the document above
2. If the question cannot be answered using this document, say "I cannot answer this question based on the uploaded document."
3. Be specific and accurate in your responses
4. Quote relevant parts of the document when appropriate
5. Do not make up information that is not in the document
6. Always reference the document name when providing information
7. If the information is not in this specific document, clearly state that"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 1,
        "stream": False
    }
    
    try:
        print(f"DEBUG: Sending RAG request to Groq API (attempt 1)")
        print(f"DEBUG: Document: {document_name}")
        print(f"DEBUG: Session: {session_id}")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        print(f"DEBUG: Response status: {response.status_code}")
        print(f"DEBUG: Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"DEBUG: Success response: {json.dumps(response_data, indent=2)}")
            return response_data["choices"][0]["message"]["content"]
        else:
            print(f"DEBUG: Error response: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Groq API error: {response.text}")
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Request timeout")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Connection error")
    except Exception as e:
        print(f"DEBUG: Exception in RAG chat completion: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling Groq API: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        session_id = chat_request.session_id
        # Only use RAG mode if there is at least one uploaded file in this session
        has_files = session_id in uploaded_files and len(uploaded_files[session_id]) > 0
        file_id = chat_request.active_file_id or active_file.get(session_id) if has_files else None

        # If no files are uploaded, clear any lingering active_file entry for this session
        if not has_files and session_id in active_file:
            del active_file[session_id]

        print(f"DEBUG: Chat request - session_id: '{session_id}', active_file_id: '{file_id}'")

        # Check if we have an active file for RAG mode and at least one file is uploaded
        if session_id and file_id and has_files:
            print(f"DEBUG: UPLOADED FILE RAG MODE - Using file_id: '{file_id}' for session: '{session_id}'")
            
            # Validate that the file belongs to the current session
            if session_id not in file_knowledge_bases:
                print(f"DEBUG: UPLOADED FILE RAG MODE - No knowledge bases found for session: '{session_id}'")
                return ChatResponse(response="No documents found for this session. Please upload a document first.")
            
            if file_id not in file_knowledge_bases[session_id]:
                print(f"DEBUG: UPLOADED FILE RAG MODE - File '{file_id}' not found in session '{session_id}'")
                return ChatResponse(response="Selected document not found in this session. Please select a valid document.")
            
            # RAG MODE: Use file-specific knowledge base
            kb = file_knowledge_bases[session_id][file_id]
            if not kb or not kb.get('chunks'):
                print(f"DEBUG: UPLOADED FILE RAG MODE - Knowledge base not found for file_id: '{file_id}'")
                return ChatResponse(response="Active file knowledge base not found.")
            
            print(f"DEBUG: UPLOADED FILE RAG MODE - Found knowledge base with {len(kb['chunks'])} chunks")
            print(f"DEBUG: UPLOADED FILE RAG MODE - Document: {kb.get('filename', 'Unknown')}")
            
            user_query = chat_request.message.strip()
            query_emb = compute_embedding(user_query)
            best_score = -1
            best_chunk = ""
            
            # Search only within the current user's uploaded document
            for chunk, emb in kb['chunks']:
                score = cosine_similarity(query_emb, emb)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
            
            print(f"DEBUG: UPLOADED FILE RAG MODE - Best similarity score: {best_score}")
            
            if best_chunk and best_score > 0.3:  # Threshold for relevance
                print(f"DEBUG: UPLOADED FILE RAG MODE - Using chunk for response (score: {best_score})")
                print(f"DEBUG: UPLOADED FILE RAG MODE - Document context: {kb.get('filename', 'Unknown')}")
                
                # Use the isolated RAG function that only uses the specific file content
                document_name = kb.get('filename', 'the uploaded document')
                response = await asyncio.to_thread(rag_chat_completion, user_query, best_chunk, session_id, document_name)
                return ChatResponse(response=response)
            else:
                print(f"DEBUG: UPLOADED FILE RAG MODE - No relevant chunk found (best score: {best_score})")
                doc_name = kb.get('filename', 'the uploaded document')
                response = f"I couldn't find relevant information in '{doc_name}' to answer your question. Please try rephrasing your question or ask about something else related to the document content."
                return ChatResponse(response=response)
        
        else:
            print(f"DEBUG: GENERAL MODE - No active file for session: '{session_id}' or no files uploaded")
            # NEW: Implement static knowledge base RAG here
            best_static_chunk_text = ""
            if static_kb:
                user_query = chat_request.message.strip()
                query_emb = compute_embedding(user_query)
                best_score = -1
                best_chunk = None

                # Search within the pre-loaded static knowledge base
                for entry in static_kb:
                    # Note: entry['embedding'] is a list, so convert it to a NumPy array for computation
                    embedding_np = np.array(entry['embedding'])
                    score = cosine_similarity(query_emb, embedding_np)
                    if score > best_score:
                        best_score = score
                        best_chunk = entry
                
                print(f"DEBUG: STATIC KB RAG - Best similarity score: {best_score}")

                # Use the best chunk if it's relevant enough
                if best_chunk and best_score > 0.4:  # Adjust this threshold as needed
                    best_static_chunk_text = best_chunk['text_content']
                    print(f"DEBUG: STATIC KB RAG - Found relevant static context for region: {best_chunk['region']}, suburb: {best_chunk['suburb']}")
            
            # GENERAL MODE: Use Groq chat completion with general knowledge
            # Get current language preference
            current_language = user_language_preference["language"]
            
            # Check if this is the first message
            is_first_message = conversation_state["first_message"]
            
            # If this is the first message, allow introduction
            if is_first_message:
                ai_response = await asyncio.to_thread(groq_chat_completion, chat_request.message, "en", is_first_message, None, session_id, best_static_chunk_text)
                conversation_state["first_message"] = False
                add_to_memory('assistant', ai_response)
                return ChatResponse(response=ai_response)
            
            # For subsequent messages, use general chat
            property_identification = identify_property_strictly(chat_request.message)
            
            # Update property tracker if a specific property is mentioned
            primary_address = property_identification["primary_address"]
            primary_locality = property_identification["primary_locality"]
            
            if primary_address:
                update_property_tracker(primary_address)
            elif primary_locality:
                update_property_tracker(f"Area: {primary_locality}")
            
            # Get current property context
            property_context = get_property_context()
            
            # Add original user message to memory
            add_to_memory('user', chat_request.message)
            
            # Check if in buyer qualification flow
            if conversation_memory["buyer_qualification"]['active']:
                response = handle_buyer_qualification(chat_request.message)
                add_to_memory('assistant', response)
                return ChatResponse(response=response)
            
            # Extract intent and preferences
            intent = extract_user_intent(chat_request.message)
            
            # Update user preferences
            if intent['preferences']:
                update_user_preferences(intent['preferences'])
            
            # Manual trigger for buyer qualification
            if chat_request.message.strip().lower() in ["start buyer qualification", "qualify me as a buyer", "i want to buy a house", "help me buy a property"]:
                response = start_buyer_qualification()
                add_to_memory('assistant', response)
                return ChatResponse(response=response)
            
            # Automatic trigger for buyer qualification if buyer intent detected
            if intent['type'] == 'search' and any(word in chat_request.message.lower() for word in ['buy', 'purchase', 'looking to buy', 'interested in buying']):
                if not conversation_memory["buyer_qualification"]['active']:
                    response = start_buyer_qualification()
                    add_to_memory('assistant', response)
                    return ChatResponse(response=response)
            
            # Get AI response in English with property and static context
            ai_response = await asyncio.to_thread(groq_chat_completion, chat_request.message, "en", False, property_context, session_id, best_static_chunk_text)
            
            # Format response to ensure it follows professional rules
            formatted_response = format_property_response(ai_response, property_context)
            
            # Add formatted response to memory
            add_to_memory('assistant', formatted_response)
                
            return ChatResponse(response=formatted_response)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"DEBUG: Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Server error while processing chat message.")

# Enhanced chat endpoint with property context
@app.post("/chat/enhanced", response_model=ChatResponse)
async def enhanced_chat_endpoint(enhanced_request: EnhancedChatRequest):
    try:
        # Get current language preference
        current_language = user_language_preference["language"]
        
        # Translate user message to English for AI processing if needed
        original_message = enhanced_request.message
        translated_message = original_message # No translation for now
        
        # Check if this is the first message
        is_first_message = conversation_state["first_message"]
        
        # If this is the first message, allow introduction without property validation
        if is_first_message:
            ai_response = await asyncio.to_thread(groq_chat_completion, translated_message, "en", is_first_message, None, None)
            # No translation for now
            conversation_state["first_message"] = False
            add_to_memory('assistant', ai_response)
            return ChatResponse(response=ai_response)
        
        # Try to identify the property mentioned by the user (but don't be strict)
        property_identification = identify_property_strictly(translated_message)
        
        # If explicit property address is provided in the request, use that
        if enhanced_request.property_address:
            property_identification["primary_address"] = enhanced_request.property_address
            property_identification["multiple_properties"] = False
            property_identification["needs_clarification"] = False
        
        # Update property tracker with the identified property
        primary_address = property_identification["primary_address"]
        primary_locality = property_identification["primary_locality"]
        
        if primary_address:
            update_property_tracker(primary_address, enhanced_request.property_context)
        elif primary_locality:
            # For locality-based queries, we'll track the locality
            update_property_tracker(f"Area: {primary_locality}", enhanced_request.property_context)
        
        # Get current property context
        property_context = get_property_context()
        
        # Add original user message to memory
        add_to_memory('user', original_message)
        
        # Check if in buyer qualification flow
        if conversation_memory["buyer_qualification"]['active']:
            response = handle_buyer_qualification(translated_message)
            # No translation for now
            conversation_memory["buyer_qualification"]['active'] = False # End qualification
            add_to_memory('assistant', response)
            return ChatResponse(response=response)
        
        # Extract intent and preferences
        intent = extract_user_intent(translated_message)
        
        # Update user preferences
        if intent['preferences']:
            update_user_preferences(intent['preferences'])
        
        # Manual trigger for buyer qualification
        if translated_message.strip().lower() in ["start buyer qualification", "qualify me as a buyer", "i want to buy a house", "help me buy a property"]:
            response = start_buyer_qualification()
            # No translation for now
            conversation_memory["buyer_qualification"]['active'] = True # Start qualification
            add_to_memory('assistant', response)
            return ChatResponse(response=response)
        
        # Automatic trigger for buyer qualification if buyer intent detected
        if intent['type'] == 'search' and any(word in translated_message.lower() for word in ['buy', 'purchase', 'looking to buy', 'interested in buying']):
            if not conversation_memory["buyer_qualification"]['active']:
                response = start_buyer_qualification()
                # No translation for now
                conversation_memory["buyer_qualification"]['active'] = True # Start qualification
                add_to_memory('assistant', response)
                return ChatResponse(response=response)
        
        # Get AI response in English with property context
        ai_response = await asyncio.to_thread(groq_chat_completion, translated_message, "en", is_first_message, property_context, None)
        
        # Format response to ensure it follows professional rules
        formatted_response = format_property_response(ai_response, property_context)
        
        # No translation for now
        conversation_state["first_message"] = False
            
        # Add translated response to memory
        add_to_memory('assistant', formatted_response)
            
        return ChatResponse(response=formatted_response)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"DEBUG: Enhanced chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail="Server error while processing enhanced chat message.")

# Language endpoints
@app.post("/language", response_model=LanguageResponse)
async def set_language(lang_req: LanguageRequest):
    try:
        lang = lang_req.language.strip().lower()
        if not lang or len(lang) > 10:
            return LanguageResponse(language=lang, success=False, error="Invalid language code.")
        user_language_preference["language"] = lang
        return LanguageResponse(language=lang, success=True)
    except Exception as e:
        print(f"DEBUG: Set language error: {str(e)}")
        return LanguageResponse(language="", success=False, error="Server error while setting language.")

@app.get("/language", response_model=LanguageResponse)
async def get_language():
    try:
        lang = user_language_preference["language"]
        return LanguageResponse(language=lang, success=True)
    except Exception as e:
        print(f"DEBUG: Get language error: {str(e)}")
        return LanguageResponse(language="", success=False, error="Server error while getting language.")

# Buyer qualification endpoint
@app.post("/buyer-qualification")
async def buyer_qualification_endpoint():
    """Start the buyer qualification process"""
    try:
        response = start_buyer_qualification()
        return {"response": response}
    except Exception as e:
        print(f"DEBUG: Buyer qualification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error during buyer qualification: {str(e)}")

# Conversation reset endpoint
@app.post("/conversation/reset")
async def conversation_reset_endpoint():
    """Reset conversation memory"""
    try:
        conversation_state["first_message"] = True
        conversation_memory["history"] = []
        conversation_memory["buyer_qualification"]["active"] = False
        conversation_memory["buyer_qualification"]["step"] = 0
        conversation_memory["buyer_qualification"]["answers"] = {}
        
        return {"success": True, "message": "Conversation memory reset successfully"}
    except Exception as e:
        print(f"DEBUG: Conversation reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error while resetting conversation: {str(e)}")

# User preferences endpoint
@app.get("/user-preferences")
async def get_user_preferences_endpoint():
    """Get current user preferences"""
    try:
        return conversation_memory["user_preferences"]
    except Exception as e:
        print(f"DEBUG: Get user preferences error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error while getting user preferences: {str(e)}")

# Streaming chat endpoint
@app.post("/streamchat")
async def stream_chat_endpoint(stream_req: StreamChatRequest):
    async def event_generator():
        # Get current language preference
        current_language = user_language_preference["language"]
        
        # Translate user message to English for AI processing if needed
        original_message = stream_req.message
        translated_message = original_message # No translation for now
        
        # Try to identify the property mentioned by the user (but don't be strict)
        property_identification = identify_property_strictly(translated_message)
        
        # Update property tracker with the identified property
        primary_address = property_identification["primary_address"]
        primary_locality = property_identification["primary_locality"]
        
        if primary_address:
            update_property_tracker(primary_address)
        elif primary_locality:
            # For locality-based queries, we'll track the locality
            update_property_tracker(f"Area: {primary_locality}")
        
        # Get current property context
        property_context = get_property_context()
        
        # Check if this is the first message
        is_first_message = conversation_state["first_message"]
        
        # Mark that conversation has started
        if is_first_message:
            conversation_state["first_message"] = False
            
        # Get AI response in English with property context
        ai_response = await asyncio.to_thread(groq_chat_completion, translated_message, "en", is_first_message, property_context)
        
        # Format response to ensure it follows professional rules
        formatted_response = format_property_response(ai_response, property_context)
        
        # No translation for now
        # Stream the translated response
        for chunk in formatted_response.split():
            yield json.dumps({"chunk": chunk + " "}) + "\n"
            await asyncio.sleep(0.05) # Simulate streaming delay
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- Helper Functions (need to be implemented if they don't exist) ---
def get_context_summary():
    # Placeholder function for conversation context
    return "Conversation history placeholder."

def identify_property_strictly(message: str):
    # Placeholder function for property identification
    return {"primary_address": None, "primary_locality": None, "multiple_properties": False, "needs_clarification": False}

def update_property_tracker(address: str, context: Optional[Dict] = None):
    # Placeholder function for property tracking
    pass

def get_property_context():
    # Placeholder function to retrieve property context
    return {}

def add_to_memory(role, message):
    # Placeholder function to add to conversation memory
    pass

def handle_buyer_qualification(message: str):
    # Placeholder for buyer qualification logic
    return "This is a placeholder for the buyer qualification flow."

def extract_user_intent(message: str):
    # Placeholder for intent extraction logic
    return {"type": "general", "preferences": {}}

def update_user_preferences(preferences: Dict):
    # Placeholder for updating user preferences
    pass

def start_buyer_qualification():
    # Placeholder for starting the qualification flow
    return "Let's start your buyer qualification. What is your budget?"

def format_property_response(response: str, context: Optional[Dict] = None):
    # Placeholder for response formatting
    return response

# --- File Upload Endpoints (Existing code) ---
@app.post("/upload_file", response_model=FileUploadResponse)
async def upload_file_endpoint(upload_request: FileUploadRequest):
    try:
        session_id = upload_request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
        
        file_id = str(uuid.uuid4())
        
        # Parse content based on file type
        content = upload_request.content
        if upload_request.file_type == 'csv':
            # Convert CSV to a readable format
            sio = io.StringIO(content)
            reader = csv.reader(sio)
            rows = list(reader)
            content = "\n".join([",".join(row) for row in rows])
        
        # Chunk and embed the content
        knowledge_base = chunk_and_embed(content)
        
        # Store in-memory knowledge base per session
        if session_id not in file_knowledge_bases:
            file_knowledge_bases[session_id] = {}
        file_knowledge_bases[session_id][file_id] = {
            'filename': upload_request.filename,
            'file_type': upload_request.file_type,
            'chunks': knowledge_base,
            'upload_time': datetime.now().isoformat()
        }
        
        # Also track the uploaded file for listing purposes
        if session_id not in uploaded_files:
            uploaded_files[session_id] = {}
        uploaded_files[session_id][file_id] = {
            'filename': upload_request.filename,
            'file_type': upload_request.file_type,
            'upload_time': datetime.now().isoformat()
        }
        
        active_file[session_id] = file_id
        
        return FileUploadResponse(
            success=True,
            message=f"File '{upload_request.filename}' uploaded and processed successfully. It is now active for RAG.",
            filename=upload_request.filename,
            file_id=file_id
        )
    except Exception as e:
        print(f"DEBUG: File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error during file upload: {str(e)}")

# NEW: Add the missing file endpoints that the frontend expects
@app.post("/file/upload", response_model=FileUploadResponse)
async def file_upload_endpoint(upload_request: FileUploadRequest):
    """File upload endpoint with /file/ prefix"""
    return await upload_file_endpoint(upload_request)

@app.get("/file/list/{session_id}")
async def file_list_endpoint(session_id: str):
    """Get list of files for a session"""
    try:
        if session_id not in uploaded_files:
            return {"files": [], "active_file_id": None}
        
        files_list = []
        for file_id, file_info in uploaded_files[session_id].items():
            files_list.append({
                "file_id": file_id,
                "filename": file_info['filename'],
                "file_type": file_info['file_type'],
                "upload_time": file_info['upload_time']
            })
        
        active_file_id = active_file.get(session_id)
        
        return {
            "files": files_list,
            "active_file_id": active_file_id
        }
    except Exception as e:
        print(f"DEBUG: File list error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error while listing files: {str(e)}")

@app.post("/file/active")
async def file_active_endpoint(request: Dict[str, Any]):
    """Set active file for a session"""
    try:
        session_id = request.get('session_id')
        file_id = request.get('file_id')
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Missing session_id")
        
        if file_id is None:
            # Clear active file
            if session_id in active_file:
                del active_file[session_id]
            return {"success": True, "message": "Active file cleared"}
        
        # Set active file
        if session_id not in file_knowledge_bases or file_id not in file_knowledge_bases[session_id]:
            raise HTTPException(status_code=404, detail="File not found in session")
        
        active_file[session_id] = file_id
        filename = file_knowledge_bases[session_id][file_id]['filename']
        
        return {"success": True, "message": f"Active file set to {filename}"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: Set active file error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error while setting active file: {str(e)}")

@app.delete("/file/clear/{session_id}")
async def file_clear_endpoint(session_id: str):
    """Clear all files for a session"""
    try:
        if session_id in uploaded_files:
            del uploaded_files[session_id]
        if session_id in file_knowledge_bases:
            del file_knowledge_bases[session_id]
        if session_id in active_file:
            del active_file[session_id]
        
        return {"success": True, "message": f"All files cleared for session {session_id}"}
    except Exception as e:
        print(f"DEBUG: Clear files error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error while clearing files: {str(e)}")

@app.post("/select_file", response_model=FileUploadResponse)
async def select_file_endpoint(request: Dict[str, str]):
    session_id = request.get('session_id')
    file_id = request.get('file_id')
    
    if not session_id or not file_id:
        raise HTTPException(status_code=400, detail="Missing session_id or file_id")
    
    if session_id not in file_knowledge_bases or file_id not in file_knowledge_bases[session_id]:
        raise HTTPException(status_code=404, detail="File not found in session knowledge base.")
        
    active_file[session_id] = file_id
    filename = file_knowledge_bases[session_id][file_id]['filename']
    
    return FileUploadResponse(
        success=True,
        message=f"File '{filename}' is now the active document for RAG.",
        filename=filename,
        file_id=file_id
    )

@app.get("/list_files", response_model=List[FileInfo])
async def list_files_endpoint(session_id: str):
    if session_id not in uploaded_files:
        return []
    
    files_list = []
    for file_id, file_info in uploaded_files[session_id].items():
        content = file_knowledge_bases[session_id][file_id]['chunks'][0][0] if file_knowledge_bases[session_id].get(file_id) and file_knowledge_bases[session_id][file_id].get('chunks') else ""
        files_list.append(FileInfo(
            filename=file_info['filename'],
            file_type=file_info['file_type'],
            content_preview=content[:50] + "..." if len(content) > 50 else content,
            upload_time=file_info['upload_time']
        ))
    return files_list

@app.post("/reset_session", response_model=ResetResponse)
async def reset_session_endpoint(request: Dict[str, str]):
    session_id = request.get('session_id')
    if session_id:
        if session_id in uploaded_files:
            del uploaded_files[session_id]
        if session_id in file_knowledge_bases:
            del file_knowledge_bases[session_id]
        if session_id in active_file:
            del active_file[session_id]
    
    conversation_state["first_message"] = True
    # Reset other in-memory state variables if necessary
    
    return ResetResponse(message="Session has been reset.")