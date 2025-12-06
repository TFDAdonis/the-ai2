import streamlit as st
import os
import requests
from pathlib import Path
import json
import concurrent.futures
from datetime import datetime

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/tfdtfd/khisbagis23/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"

# Enhanced persona with deep thinking capability
PRESET_PROMPTS = {
    "Khisba GIS": """You are Khisba GIS, an enthusiastic remote sensing and GIS expert with enhanced analytical capabilities. Your personality:
- Name: Khisba GIS
- Role: Remote sensing and GIS expert with deep thinking
- Style: Warm, friendly, approachable, and deeply analytical
- Expertise: Deep knowledge of satellite imagery, vegetation indices, and geospatial analysis
- Thinking Process: 
  1. First analyze the problem/question carefully
  2. Consider multiple perspectives
  3. Search for relevant information
  4. Synthesize findings
  5. Provide well-reasoned conclusions
  6. Suggest next steps or additional research
- Always eager to explore complex remote sensing challenges

Guidelines:
- Use Chain-of-Thought reasoning in your responses
- When presented with search results, analyze them critically
- Identify connections between different pieces of information
- Highlight contradictions or gaps in knowledge
- Provide actionable insights and recommendations
- When unsure, acknowledge limitations and suggest how to find answers
- Always introduce yourself as Khisba GIS when asked who you are""",
    
    "Deep Thinker": """You are an AI with enhanced analytical capabilities. Your thinking process follows these steps:
1. PROBLEM UNDERSTANDING: Carefully analyze what's being asked
2. PERSPECTIVE TAKING: Consider the question from multiple angles
3. INFORMATION GATHERING: Use available search tools to find relevant data
4. CRITICAL ANALYSIS: Evaluate information credibility, look for patterns
5. SYNTHESIS: Combine information from different sources
6. REASONING: Apply logical deduction and inference
7. CONCLUSIONS: Present well-supported conclusions
8. SUGGESTIONS: Offer next steps or additional questions

Always structure your responses to show your thinking process. Be honest about uncertainty.""",
    
    "Research Assistant": """You are a professional research assistant. You:
1. Systematically search across all available sources
2. Compare and contrast findings from different databases
3. Note conflicts or agreements between sources
4. Highlight the most reliable information
5. Summarize key findings concisely
6. Suggest additional avenues for investigation
7. Always cite your sources when possible""",
    
    "Default Assistant": "You are a helpful, friendly AI assistant. Provide clear and concise answers.",
    "Custom": ""
}

# Search APIs - Expanded with 16 sources
SEARCH_TOOLS = {
    "ArXiv": {
        "name": "ArXiv Scientific Papers",
        "icon": "📚",
        "description": "Search scientific papers",
        "endpoint": "http://export.arxiv.org/api/query"
    },
    "DuckDuckGo": {
        "name": "DuckDuckGo Web Search",
        "icon": "🔍",
        "description": "Search the web",
        "endpoint": "https://api.duckduckgo.com/"
    },
    "Weather": {
        "name": "Weather",
        "icon": "🌤️",
        "description": "Get weather information",
        "endpoint": "https://wttr.in/"
    },
    "Wikipedia": {
        "name": "Wikipedia",
        "icon": "📖",
        "description": "Search Wikipedia",
        "endpoint": "https://en.wikipedia.org/w/api.php"
    },
    "GitHub": {
        "name": "GitHub Repositories",
        "icon": "💻",
        "description": "Search GitHub repositories",
        "endpoint": "https://api.github.com/search/repositories"
    },
    "StackOverflow": {
        "name": "Stack Overflow",
        "icon": "🔧",
        "description": "Search programming questions",
        "endpoint": "https://api.stackexchange.com/2.3/search"
    },
    "OpenLibrary": {
        "name": "Books",
        "icon": "📖",
        "description": "Search books",
        "endpoint": "https://openlibrary.org/search.json"
    },
    "Dictionary": {
        "name": "Dictionary",
        "icon": "📖",
        "description": "Word definitions",
        "endpoint": "https://api.dictionaryapi.dev/api/v2/entries/en/"
    },
    "Countries": {
        "name": "Countries",
        "icon": "🌍",
        "description": "Country information",
        "endpoint": "https://restcountries.com/v3.1/"
    },
    "Quotes": {
        "name": "Quotes",
        "icon": "💬",
        "description": "Search quotes",
        "endpoint": "https://api.quotable.io/search/quotes"
    },
    "PubMed": {
        "name": "PubMed",
        "icon": "🏥",
        "description": "Medical research",
        "endpoint": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    },
    "AirQuality": {
        "name": "Air Quality",
        "icon": "🌬️",
        "description": "Air quality data",
        "endpoint": "https://api.openaq.org/v2/"
    },
    "Geocoding": {
        "name": "Geocoding",
        "icon": "📍",
        "description": "Location coordinates",
        "endpoint": "https://nominatim.openstreetmap.org/search"
    },
    "Wikidata": {
        "name": "Wikidata",
        "icon": "🗃️",
        "description": "Structured data",
        "endpoint": "https://www.wikidata.org/w/api.php"
    }
}

st.set_page_config(
    page_title="DeepThink LLAMA",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 DeepThink LLAMA")
st.caption("A smart AI that thinks deeply and searches across multiple sources")

# Enhanced download function
def download_model():
    MODEL_DIR.mkdir(exist_ok=True)
    st.info("📥 Downloading TinyLLaMA model...")
    
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download model: {str(e)}")
    
    total_size = int(response.headers.get('content-length', 0))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    downloaded = 0
    try:
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {downloaded / (1024**2):.1f} / {total_size / (1024**2):.1f} MB")
    except Exception as e:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise Exception(f"Download interrupted: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    if MODEL_PATH.exists():
        file_size = MODEL_PATH.stat().st_size / (1024**3)
        st.success(f"✅ Download successful! File size: {file_size:.2f} GB")
        return True
    else:
        raise Exception("❌ Download failed")

@st.cache_resource(show_spinner=False)
def load_model():
    from ctransformers import AutoModelForCausalLM
    
    if not MODEL_PATH.exists():
        download_model()
    
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        model_file=MODEL_PATH.name,
        model_type="llama",
        context_length=4096,
        gpu_layers=0,
        threads=8
    )
    return model

# Enhanced search functions with error handling
def search_arxiv(query, max_results=3):
    """Search ArXiv for scientific papers."""
    try:
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        response = requests.get(SEARCH_TOOLS["ArXiv"]["endpoint"], params=params, timeout=10)
        response.raise_for_status()
        
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        results = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            authors = [author.find('{http://www.w3.org/2005/Atom}name').text 
                      for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            
            results.append({
                'title': title,
                'summary': summary[:300] + '...' if len(summary) > 300 else summary,
                'authors': ', '.join(authors[:3]),
                'published': published[:10] if published else 'N/A',
                'type': 'scientific_paper',
                'source': 'ArXiv'
            })
        return results if results else [{"info": "No papers found", "query": query}]
    except Exception as e:
        return [{"error": f"ArXiv search error: {str(e)}", "query": query}]

def search_wikipedia(query):
    """Search Wikipedia for articles."""
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 3,
            'utf8': 1
        }
        response = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('query', {}).get('search', []):
            # Get page summary
            params2 = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info',
                'inprop': 'url',
                'exintro': 1,
                'explaintext': 1,
                'pageids': item['pageid']
            }
            response2 = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params2, timeout=10)
            if response2.status_code == 200:
                page_data = response2.json()
                pages = page_data.get('query', {}).get('pages', {})
                for page_id, page_info in pages.items():
                    results.append({
                        'title': page_info.get('title', ''),
                        'summary': page_info.get('extract', '')[:400] + '...',
                        'url': page_info.get('fullurl', ''),
                        'type': 'encyclopedia',
                        'source': 'Wikipedia'
                    })
        
        return results if results else [{"info": "No Wikipedia articles found", "query": query}]
    except Exception as e:
        return [{"error": f"Wikipedia search error: {str(e)}", "query": query}]

def search_github(query, max_results=3):
    """Search GitHub repositories."""
    try:
        params = {
            'q': query,
            'sort': 'stars',
            'order': 'desc',
            'per_page': max_results
        }
        headers = {'Accept': 'application/vnd.github.v3+json'}
        response = requests.get(SEARCH_TOOLS["GitHub"]["endpoint"], 
                              params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for repo in data.get('items', [])[:max_results]:
            results.append({
                'name': repo.get('full_name', ''),
                'description': repo.get('description', 'No description'),
                'stars': repo.get('stargazers_count', 0),
                'language': repo.get('language', 'N/A'),
                'url': repo.get('html_url', ''),
                'type': 'code_repository',
                'source': 'GitHub'
            })
        return results if results else [{"info": "No repositories found", "query": query}]
    except Exception as e:
        return [{"error": f"GitHub search error: {str(e)}", "query": query}]

def get_weather(location="London"):
    """Get weather information."""
    try:
        response = requests.get(f"{SEARCH_TOOLS['Weather']['endpoint']}/{location}?format=j1", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'current_condition' in data:
            current = data['current_condition'][0]
            return {
                'location': location,
                'temperature_c': current.get('temp_C', 'N/A'),
                'temperature_f': current.get('temp_F', 'N/A'),
                'condition': current.get('weatherDesc', [{}])[0].get('value', 'N/A'),
                'humidity': current.get('humidity', 'N/A'),
                'wind_speed': current.get('windspeedKmph', 'N/A'),
                'type': 'weather',
                'source': 'wttr.in'
            }
        return {"info": "No weather data found", "query": location}
    except Exception as e:
        return {"error": f"Weather lookup error: {str(e)}", "query": location}

def search_duckduckgo(query):
    """Search DuckDuckGo for instant answers."""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }
        response = requests.get(SEARCH_TOOLS["DuckDuckGo"]["endpoint"], params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        result = {
            'abstract': data.get('AbstractText', ''),
            'answer': data.get('Answer', ''),
            'related': [topic.get('Text', '') for topic in data.get('RelatedTopics', [])[:3]],
            'type': 'web_search',
            'source': 'DuckDuckGo'
        }
        
        # Clean empty values
        result = {k: v for k, v in result.items() if v}
        return result if any(result.values()) else {"info": "No instant answer found", "query": query}
    except Exception as e:
        return {"error": f"DuckDuckGo search error: {str(e)}", "query": query}

def search_openlibrary(query, max_results=3):
    """Search OpenLibrary for books."""
    try:
        params = {
            'q': query,
            'limit': max_results,
            'fields': 'title,author_name,first_publish_year,subject'
        }
        response = requests.get(SEARCH_TOOLS["OpenLibrary"]["endpoint"], params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for doc in data.get('docs', [])[:max_results]:
            results.append({
                'title': doc.get('title', 'Unknown Title'),
                'authors': doc.get('author_name', ['Unknown Author'])[:2],
                'year': doc.get('first_publish_year', 'N/A'),
                'subjects': doc.get('subject', [])[:3],
                'type': 'book',
                'source': 'OpenLibrary'
            })
        return results if results else [{"info": "No books found", "query": query}]
    except Exception as e:
        return [{"error": f"OpenLibrary search error: {str(e)}", "query": query}]

def search_country(query):
    """Search for country information."""
    try:
        response = requests.get(f"{SEARCH_TOOLS['Countries']['endpoint']}/name/{query}", timeout=10)
        if response.status_code == 404:
            return {"info": "Country not found", "query": query}
        response.raise_for_status()
        data = response.json()
        
        if data:
            country = data[0]
            return {
                'name': country.get('name', {}).get('common', 'Unknown'),
                'official_name': country.get('name', {}).get('official', 'Unknown'),
                'capital': ', '.join(country.get('capital', ['N/A'])),
                'population': country.get('population', 'N/A'),
                'region': country.get('region', 'N/A'),
                'subregion': country.get('subregion', 'N/A'),
                'languages': list(country.get('languages', {}).values())[:3] if country.get('languages') else [],
                'currencies': list(country.get('currencies', {}).keys())[:2] if country.get('currencies') else [],
                'flag_emoji': country.get('flag', ''),
                'type': 'country_info',
                'source': 'REST Countries'
            }
        return {"info": "No country data found", "query": query}
    except Exception as e:
        return {"error": f"Country search error: {str(e)}", "query": query}

def search_all_relevant(query, max_sources=6):
    """Search multiple relevant sources based on query type."""
    search_functions = []
    
    # Determine which sources to search based on query
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['weather', 'temperature', 'forecast', 'climate']):
        search_functions.append(('weather', lambda: get_weather(query)))
    
    if any(word in query_lower for word in ['paper', 'research', 'study', 'arxiv', 'science', 'academic']):
        search_functions.append(('arxiv', lambda: search_arxiv(query, 2)))
    
    if any(word in query_lower for word in ['define', 'definition', 'meaning', 'word', 'vocabulary']):
        # We'll use Wikipedia as dictionary fallback
        search_functions.append(('dictionary', lambda: [{"word": query.split()[0], "definition": "Check dictionary for precise meaning", "source": "note"}]))
    
    if any(word in query_lower for word in ['code', 'github', 'repository', 'programming', 'software']):
        search_functions.append(('github', lambda: search_github(query, 2)))
    
    if any(word in query_lower for word in ['country', 'capital', 'population', 'flag', 'nation']):
        search_functions.append(('countries', lambda: [search_country(query)] if isinstance(search_country(query), dict) else search_country(query)))
    
    if any(word in query_lower for word in ['book', 'author', 'novel', 'literature', 'read']):
        search_functions.append(('books', lambda: search_openlibrary(query, 2)))
    
    # Always include these core sources
    search_functions.append(('wikipedia', lambda: search_wikipedia(query)))
    search_functions.append(('duckduckgo', lambda: search_duckduckgo(query)))
    
    # Limit to max_sources
    search_functions = search_functions[:max_sources]
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(search_functions)) as executor:
        future_to_name = {executor.submit(func): name for name, func in search_functions}
        
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = {"error": str(e), "query": query}
    
    return results

def format_search_summary(query, search_results):
    """Create a concise summary of search results."""
    summary_parts = []
    
    for source, data in search_results.items():
        if isinstance(data, dict) and 'error' not in data and data:
            if source == 'duckduckgo' and data.get('answer'):
                summary_parts.append(f"💡 Quick answer: {data['answer']}")
            elif source == 'wikipedia' and isinstance(data, list) and data:
                for item in data[:1]:
                    if isinstance(item, dict) and 'title' in item:
                        summary_parts.append(f"📚 Wikipedia: {item.get('title', '')} - {item.get('summary', '')[:150]}...")
            elif source == 'arxiv' and isinstance(data, list) and data:
                for item in data[:1]:
                    if isinstance(item, dict) and 'title' in item:
                        summary_parts.append(f"🔬 Research: {item.get('title', '')[:100]}...")
            elif source == 'weather' and isinstance(data, dict) and data.get('temperature_c'):
                summary_parts.append(f"🌤️ Weather: {data.get('temperature_c')}°C, {data.get('condition', '')}")
            elif source == 'countries' and isinstance(data, list) and data:
                for item in data[:1]:
                    if isinstance(item, dict) and 'name' in item:
                        summary_parts.append(f"🌍 Country: {item.get('name', '')}")
    
    if summary_parts:
        return "\n".join(summary_parts)
    return ""

def format_prompt_with_thinking(messages, system_prompt, search_context=""):
    """Format prompt with chain-of-thought reasoning."""
    prompt = f"""<|system|>
{system_prompt}

Current date: {datetime.now().strftime('%Y-%m-%d')}

THINKING FRAMEWORK:
1. Understand the user's query deeply
2. Consider what information is needed
3. Search relevant sources (if applicable)
4. Analyze and synthesize information
5. Provide clear, reasoned response
6. Suggest next steps or additional questions

{search_context if search_context else "No search context provided. Rely on general knowledge."}
</s>
"""
    
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['content']}</s>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['content']}</s>\n"
    
    prompt += "<|assistant|>\nLet me think through this carefully...\n\n"
    return prompt

def generate_deep_response(model, messages, system_prompt, search_results=None, max_tokens=512, temperature=0.7):
    """Generate a thoughtful response with analysis."""
    # Prepare search context
    search_context = ""
    if search_results:
        search_context = "SEARCH RESULTS SUMMARY:\n"
        for source, data in search_results.items():
            if data and 'error' not in str(data):
                search_context += f"\n{source.upper()}:\n"
                if isinstance(data, list):
                    for item in data[:2]:
                        if isinstance(item, dict):
                            for key, value in item.items():
                                if key not in ['error', 'type', 'source'] and value:
                                    search_context += f"- {key}: {str(value)[:100]}\n"
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if key not in ['error', 'type', 'source'] and value:
                            search_context += f"- {key}: {str(value)[:100]}\n"
    
    # Format prompt with thinking framework
    prompt = format_prompt_with_thinking(messages, system_prompt, search_context)
    
    # Generate response
    response = model(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        stop=["</s>", "<|user|>", "<|assistant|>", "<|system|>", "\n\n", "###"]
    )
    
    # Clean up response
    response = response.strip()
    
    # Ensure response shows thinking process
    if "Let me think through this carefully..." in response:
        response = response.replace("Let me think through this carefully...", "")
    
    # Add source citations if available
    if search_results:
        sources_used = [source for source in search_results if search_results[source] and 'error' not in str(search_results[source])]
        if sources_used:
            response += f"\n\n📚 *Sources consulted: {', '.join(sources_used)}*"
    
    return response

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = PRESET_PROMPTS["Deep Thinker"]

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Deep Thinker"

if "search_history" not in st.session_state:
    st.session_state.search_history = []

# Sidebar
with st.sidebar:
    st.header("🧠 Thinking Mode")
    
    selected_preset = st.selectbox(
        "Choose thinking style:",
        options=list(PRESET_PROMPTS.keys()),
        index=list(PRESET_PROMPTS.keys()).index(st.session_state.selected_preset),
        key="preset_selector"
    )
    
    if selected_preset != st.session_state.selected_preset:
        st.session_state.selected_preset = selected_preset
        if selected_preset != "Custom":
            st.session_state.system_prompt = PRESET_PROMPTS[selected_preset]
    
    system_prompt = st.text_area(
        "Thinking instructions:",
        value=st.session_state.system_prompt,
        height=200,
        key="system_prompt_input"
    )
    
    if system_prompt != st.session_state.system_prompt:
        st.session_state.system_prompt = system_prompt
        if system_prompt not in PRESET_PROMPTS.values():
            st.session_state.selected_preset = "Custom"
    
    st.divider()
    
    st.header("🔍 Smart Search")
    st.caption("Automatically searches relevant sources")
    
    search_mode = st.radio(
        "Search strategy:",
        ["Smart (auto-select sources)", "Comprehensive (all sources)", "Focused (few sources)"],
        index=0
    )
    
    auto_search = st.checkbox("Auto-search before responding", value=True)
    
    st.divider()
    
    st.header("⚙️ Thinking Parameters")
    
    thinking_depth = st.select_slider(
        "Thinking depth:",
        options=["Quick", "Balanced", "Deep", "Very Deep"],
        value="Balanced"
    )
    
    temperature = st.slider(
        "Creativity:", 
        0.1, 2.0, 0.7, 0.1,
        help="Higher = more creative, Lower = more focused"
    )
    
    max_tokens = st.slider(
        "Response length:", 
        128, 1024, 512, 64
    )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.system_prompt = PRESET_PROMPTS["Deep Thinker"]
            st.session_state.selected_preset = "Deep Thinker"
            st.rerun()
    
    st.divider()
    st.caption("🧠 DeepThink LLAMA v1.0")
    st.caption("Combines reasoning with multi-source search")

# Main area
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🧠 DeepThink LLAMA")
    st.caption("An AI that thinks before responding and searches smartly")
with col2:
    if auto_search:
        st.info("🔍 Auto-search enabled")

# Display current thinking style
with st.expander("Current Thinking Style", expanded=False):
    st.info(st.session_state.system_prompt[:500] + "..." if len(st.session_state.system_prompt) > 500 else st.session_state.system_prompt)

# Load model
if not st.session_state.model_loaded:
    with st.spinner("🧠 Loading DeepThink LLAMA... This may take a moment on first run."):
        try:
            model = load_model()
            st.session_state.model_loaded = True
            st.session_state.model = model
            st.success("✅ Model loaded and ready for deep thinking")
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything... I'll think deeply about it"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare for response
    with st.chat_message("assistant"):
        # Show thinking status
        thinking_container = st.empty()
        search_container = st.empty()
        response_container = st.empty()
        
        # Step 1: Show thinking
        thinking_container.info("💭 Analyzing your question and thinking deeply...")
        
        # Step 2: Determine search strategy
        search_results = None
        if auto_search:
            thinking_container.info("🔍 Searching relevant information sources...")
            
            # Determine search strategy
            if search_mode == "Comprehensive":
                max_sources = 8
            elif search_mode == "Focused":
                max_sources = 3
            else:  # Smart
                max_sources = 6
            
            with st.spinner(f"Searching {max_sources} relevant sources..."):
                search_results = search_all_relevant(prompt, max_sources)
            
            # Display search summary
            if search_results:
                with st.expander("📊 Search Summary", expanded=False):
                    for source, data in search_results.items():
                        if data and 'error' not in str(data):
                            st.write(f"**{source.title()}:**")
                            if isinstance(data, list):
                                for item in data[:2]:
                                    if isinstance(item, dict):
                                        # Show key information
                                        keys_to_show = ['title', 'name', 'summary', 'answer', 'temperature_c']
                                        for key in keys_to_show:
                                            if key in item and item[key]:
                                                st.write(f"- {key}: {str(item[key])[:150]}")
                                                break
                                        else:
                                            # Show first non-error key
                                            for key, value in item.items():
                                                if key not in ['error', 'type', 'source'] and value:
                                                    st.write(f"- {key}: {str(value)[:100]}")
                                                    break
                            elif isinstance(data, dict):
                                for key, value in data.items():
                                    if key not in ['error', 'type', 'source'] and value:
                                        st.write(f"- {key}: {str(value)[:150]}")
        
        # Step 3: Generate thoughtful response
        thinking_container.info("🤔 Synthesizing information and formulating response...")
        
        with st.spinner("Formulating thoughtful response..."):
            # Adjust parameters based on thinking depth
            if thinking_depth == "Quick":
                response_tokens = 256
            elif thinking_depth == "Balanced":
                response_tokens = 512
            elif thinking_depth == "Deep":
                response_tokens = 768
            else:  # Very Deep
                response_tokens = 1024
            
            response = generate_deep_response(
                st.session_state.model,
                st.session_state.messages,
                system_prompt=st.session_state.system_prompt,
                search_results=search_results,
                max_tokens=response_tokens,
                temperature=temperature
            )
        
        # Display response
        thinking_container.empty()
        response_container.markdown(response)
        
        # Add suggestions based on thinking depth
        if thinking_depth in ["Deep", "Very Deep"]:
            with st.expander("💡 Suggestions for further exploration"):
                suggestions = """
                1. **Deeper research questions:**
                   - What are the underlying assumptions?
                   - What counterarguments exist?
                   - What are the practical implications?
                
                2. **Additional data sources to consider:**
                   - Academic databases
                   - Official statistics
                   - Expert interviews
                
                3. **Next steps:**
                   - Verify with primary sources
                   - Consider different perspectives
                   - Test with real-world examples
                """
                st.markdown(suggestions)
    
    # Add to conversation history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Store search in history
    if search_results:
        st.session_state.search_history.append({
            "query": prompt,
            "results": search_results,
            "timestamp": datetime.now().isoformat()
        })

# Search history viewer
if st.session_state.search_history:
    with st.sidebar.expander("📚 Search History", expanded=False):
        for i, search in enumerate(st.session_state.search_history[-5:]):
            st.caption(f"**{search['query'][:50]}...**")
            st.caption(f"_{search['timestamp'][:16]}_")
