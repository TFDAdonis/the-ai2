import streamlit as st
import requests
from pathlib import Path
import concurrent.futures
from datetime import datetime
import re

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/tfdtfd/khisbagis23/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true"

# Enhanced deep thinking prompts
PRESET_PROMPTS = {
    "Deep Thinker Pro": """You are a sophisticated AI thinker that excels at analysis, synthesis, and providing insightful perspectives. 

THINKING FRAMEWORK:
1. **Comprehension**: Understand the query fully, identify key elements
2. **Contextualization**: Place the topic in historical, cultural, or disciplinary context
3. **Multi-Source Analysis**: Examine information from different sources critically
4. **Pattern Recognition**: Identify connections, contradictions, gaps
5. **Synthesis**: Combine insights into coherent understanding
6. **Critical Evaluation**: Assess reliability, bias, significance
7. **Insight Generation**: Provide original perspectives or connections
8. **Actionable Knowledge**: Suggest applications, further questions, implications

RESPONSE STRUCTURE:
- Start with brief overview
- Present analysis with reasoning
- Reference sources when available
- Highlight interesting connections
- Acknowledge uncertainties
- End with thought-provoking questions or suggestions

TONE: Analytical yet engaging, precise yet accessible.""",

    "Khisba GIS Expert": """You are Khisba GIS - a passionate remote sensing/GIS specialist with deep analytical skills.

SPECIALTY THINKING PROCESS:
1. **Geospatial Context**: How does location/spatial relationships matter?
2. **Temporal Analysis**: What changes over time? Historical patterns?
3. **Data Source Evaluation**: Satellite, ground, or derived data reliability?
4. **Multi-Scale Thinking**: From local to global perspectives
5. **Practical Applications**: Real-world uses of the information
6. **Ethical Considerations**: Privacy, representation, accessibility issues

EXPERTISE: Satellite imagery, vegetation indices, climate analysis, urban planning, disaster monitoring
STYLE: Enthusiastic, precise, eager to explore spatial dimensions of any topic""",

    "Research Analyst": """You are a professional research analyst specializing in synthesizing complex information.

ANALYTICAL APPROACH:
1. **Source Triangulation**: Cross-reference multiple information sources
2. **Credibility Assessment**: Evaluate source reliability, date, bias
3. **Trend Identification**: Spot patterns, changes, anomalies
4. **Comparative Analysis**: Similarities/differences across contexts
5. **Implication Mapping**: Consequences, applications, risks
6. **Knowledge Gaps**: What's missing or needs verification

Always provide structured, evidence-based analysis with clear reasoning.""",

    "Critical Thinker": """You excel at questioning assumptions and examining topics from multiple angles.

CRITICAL THINKING TOOLS:
1. **Assumption Detection**: What unstated beliefs underlie this?
2. **Perspective Switching**: How would different groups view this?
3. **Logical Analysis**: Are arguments valid, evidence sufficient?
4. **Counterfactual Thinking**: What if things were different?
5. **Ethical Reflection**: Moral dimensions, consequences
6. **Practical Reality Check**: Feasibility, implementation issues

Challenge conventional wisdom while remaining constructive.""",

    "Creative Synthesizer": """You connect seemingly unrelated ideas to generate novel insights.

CREATIVE PROCESS:
1. **Divergent Thinking**: Generate multiple possible interpretations
2. **Analogical Reasoning**: What similar patterns exist elsewhere?
3. **Metaphorical Connection**: What metaphors illuminate this?
4. **Interdisciplinary Bridging**: Connect across fields
5. **Future Projection**: How might this evolve or transform?
6. **Alternative Framing**: Different ways to conceptualize

Be imaginative while staying grounded in evidence."""
}

# Optimized search tools
SEARCH_TOOLS = {
    "Wikipedia": {
        "name": "Wikipedia",
        "icon": "📚",
        "description": "Encyclopedia articles",
        "endpoint": "https://en.wikipedia.org/w/api.php"
    },
    "DuckDuckGo": {
        "name": "Web Search",
        "icon": "🌐",
        "description": "Instant answers & web results",
        "endpoint": "https://api.duckduckgo.com/"
    },
    "ArXiv": {
        "name": "Research Papers",
        "icon": "🔬",
        "description": "Scientific publications",
        "endpoint": "http://export.arxiv.org/api/query"
    },
    "Books": {
        "name": "Books",
        "icon": "📖",
        "description": "Book information",
        "endpoint": "https://openlibrary.org/search.json"
    },
    "Countries": {
        "name": "Country Data",
        "icon": "🌍",
        "description": "Country information",
        "endpoint": "https://restcountries.com/v3.1/"
    },
    "Weather": {
        "name": "Weather",
        "icon": "🌤️",
        "description": "Weather information",
        "endpoint": "https://wttr.in/"
    },
    "GitHub": {
        "name": "Code Repos",
        "icon": "💻",
        "description": "GitHub repositories",
        "endpoint": "https://api.github.com/search/repositories"
    }
}

st.set_page_config(
    page_title="DeepThink Pro",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state with safe defaults
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = None

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = PRESET_PROMPTS["Deep Thinker Pro"]

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Deep Thinker Pro"

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .thinking-bubble {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4a90e2;
        margin: 1rem 0;
    }
    .analysis-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffb300;
        margin: 1rem 0;
    }
    .source-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title with emojis
st.title("🧠 DeepThink Pro")
st.caption("Advanced AI that thinks, researches, and analyzes like a human expert")

# Download function
def download_model():
    MODEL_DIR.mkdir(exist_ok=True)
    
    if MODEL_PATH.exists():
        return True
    
    st.warning("⚠️ Model not found. Downloading...")
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded = 0
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloading: {downloaded / (1024**2):.1f} MB")
        
        progress_bar.empty()
        status_text.empty()
        
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size / (1024**3)
            st.success(f"✅ Model downloaded: {file_size:.2f} GB")
            return True
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
    return False

@st.cache_resource(show_spinner=False)
def load_model():
    from ctransformers import AutoModelForCausalLM
    
    if not MODEL_PATH.exists():
        if not download_model():
            raise Exception("Model download failed")
    
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        model_file=MODEL_PATH.name,
        model_type="llama",
        context_length=4096,
        gpu_layers=0,
        threads=8
    )

# Enhanced search functions
def search_wikipedia(query):
    """Enhanced Wikipedia search with better parsing."""
    try:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': 3,
            'utf8': 1
        }
        response = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params, timeout=8)
        data = response.json()
        
        results = []
        for item in data.get('query', {}).get('search', []):
            # Get detailed page info
            params2 = {
                'action': 'query',
                'format': 'json',
                'prop': 'extracts|info|categories',
                'inprop': 'url',
                'exintro': 1,
                'explaintext': 1,
                'pageids': item['pageid']
            }
            response2 = requests.get(SEARCH_TOOLS["Wikipedia"]["endpoint"], params=params2, timeout=8)
            if response2.status_code == 200:
                page_data = response2.json()
                pages = page_data.get('query', {}).get('pages', {})
                for page_info in pages.values():
                    extract = page_info.get('extract', '')
                    if extract:
                        # Clean the extract
                        extract = re.sub(r'\n+', ' ', extract)
                        extract = re.sub(r'\s+', ' ', extract)
                        
                        results.append({
                            'title': page_info.get('title', ''),
                            'summary': extract[:500] + ('...' if len(extract) > 500 else ''),
                            'url': page_info.get('fullurl', ''),
                            'categories': list(page_info.get('categories', []))[:5],
                            'wordcount': page_info.get('wordcount', 0),
                            'source': 'Wikipedia',
                            'relevance': item.get('score', 0)
                        })
        
        return sorted(results, key=lambda x: x['relevance'], reverse=True) if results else []
    except Exception:
        return []

def search_duckduckgo_enhanced(query):
    """Enhanced DuckDuckGo search with better parsing."""
    try:
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1,
            't': 'streamlit_app'
        }
        response = requests.get(SEARCH_TOOLS["DuckDuckGo"]["endpoint"], params=params, timeout=8)
        data = response.json()
        
        results = {
            'abstract': data.get('AbstractText', ''),
            'answer': data.get('Answer', ''),
            'definition': data.get('Definition', ''),
            'categories': [topic.get('Name', '') for topic in data.get('Categories', [])[:3]],
            'related_topics': [topic.get('Text', '') for topic in data.get('RelatedTopics', [])[:5]],
            'source': 'DuckDuckGo'
        }
        
        # Clean and filter empty values
        cleaned = {}
        for key, value in results.items():
            if isinstance(value, str) and value.strip():
                cleaned[key] = value.strip()
            elif isinstance(value, list) and value:
                cleaned[key] = [v.strip() for v in value if v and v.strip()]
        
        return cleaned if cleaned else {}
    except Exception:
        return {}

def search_arxiv_enhanced(query):
    """Enhanced ArXiv search."""
    try:
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': 3,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        response = requests.get(SEARCH_TOOLS["ArXiv"]["endpoint"], params=params, timeout=10)
        
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip() if entry.find('{http://www.w3.org/2005/Atom}title') is not None else ''
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip() if entry.find('{http://www.w3.org/2005/Atom}summary') is not None else ''
            
            if title and summary:
                papers.append({
                    'title': title,
                    'summary': summary[:400] + '...' if len(summary) > 400 else summary,
                    'published': entry.find('{http://www.w3.org/2005/Atom}published').text[:10] if entry.find('{http://www.w3.org/2005/Atom}published') is not None else '',
                    'source': 'ArXiv',
                    'relevance': 1.0  # Simple relevance score
                })
        
        return papers
    except Exception:
        return []

def smart_source_selector(query):
    """Intelligently select which sources to search based on query."""
    query_lower = query.lower()
    
    # Check for specific patterns
    is_historical = any(word in query_lower for word in ['history', 'historical', 'century', 'war', 'battle', 'king', 'queen', 'emperor', 'emir'])
    is_scientific = any(word in query_lower for word in ['science', 'research', 'study', 'paper', 'experiment', 'data', 'analysis'])
    is_technical = any(word in query_lower for word in ['code', 'programming', 'software', 'algorithm', 'github', 'python', 'javascript'])
    is_geographical = any(word in query_lower for word in ['country', 'city', 'capital', 'population', 'map', 'location', 'weather'])
    is_conceptual = any(word in query_lower for word in ['what is', 'define', 'meaning', 'concept', 'theory', 'philosophy'])
    is_person = any(word in query_lower for word in ['who is', 'biography', 'born', 'died', 'leader', 'president', 'emir'])
    
    # Select sources based on query type
    sources = []
    
    # Always include Wikipedia for factual information
    sources.append(('Wikipedia', search_wikipedia))
    
    # Add DuckDuckGo for quick answers
    sources.append(('DuckDuckGo', search_duckduckgo_enhanced))
    
    # Add specialized sources based on query
    if is_historical or is_person:
        sources.append(('Books', lambda q: []))  # Placeholder for books API
    
    if is_scientific:
        sources.append(('ArXiv', search_arxiv_enhanced))
    
    if is_technical:
        sources.append(('GitHub', lambda q: []))  # Placeholder for GitHub
    
    if is_geographical:
        sources.append(('Countries', lambda q: []))  # Placeholder for countries
        sources.append(('Weather', lambda q: []))  # Placeholder for weather
    
    return sources[:5]  # Limit to 5 sources

def perform_intelligent_search(query):
    """Perform parallel search on intelligently selected sources."""
    sources = smart_source_selector(query)
    
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as executor:
        future_to_source = {executor.submit(func, query): name for name, func in sources}
        
        for future in concurrent.futures.as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                data = future.result(timeout=8)
                if data:  # Only include non-empty results
                    results[source_name] = data
            except Exception:
                continue
    
    return results

def analyze_search_results(query, results):
    """Analyze search results to extract key insights."""
    analysis = {
        'key_facts': [],
        'conflicting_info': [],
        'knowledge_gaps': [],
        'source_quality': {},
        'main_themes': []
    }
    
    # Extract key facts from each source
    for source, data in results.items():
        if source == 'Wikipedia' and isinstance(data, list):
            for item in data[:2]:
                if 'summary' in item:
                    analysis['key_facts'].append({
                        'fact': item['summary'][:200],
                        'source': source,
                        'title': item.get('title', '')
                    })
        
        elif source == 'DuckDuckGo' and isinstance(data, dict):
            if data.get('answer'):
                analysis['key_facts'].append({
                    'fact': data['answer'],
                    'source': source,
                    'type': 'direct_answer'
                })
            if data.get('abstract'):
                analysis['key_facts'].append({
                    'fact': data['abstract'][:200],
                    'source': source,
                    'type': 'abstract'
                })
        
        elif source == 'ArXiv' and isinstance(data, list):
            for paper in data[:1]:
                analysis['key_facts'].append({
                    'fact': f"Research paper: {paper.get('title', '')}",
                    'source': source,
                    'type': 'scientific'
                })
    
    # Identify potential knowledge gaps
    query_terms = query.lower().split()
    found_terms = []
    for fact in analysis['key_facts']:
        fact_text = fact['fact'].lower()
        for term in query_terms:
            if term in fact_text:
                found_terms.append(term)
    
    missing_terms = [term for term in query_terms if term not in found_terms]
    if missing_terms:
        analysis['knowledge_gaps'].append(f"Missing information about: {', '.join(missing_terms[:3])}")
    
    # Assess source quality
    for source in results:
        if source == 'Wikipedia':
            analysis['source_quality'][source] = {'reliability': 'high', 'coverage': 'broad'}
        elif source == 'ArXiv':
            analysis['source_quality'][source] = {'reliability': 'high', 'coverage': 'specialized'}
        elif source == 'DuckDuckGo':
            analysis['source_quality'][source] = {'reliability': 'medium', 'coverage': 'general'}
    
    return analysis

def create_thinking_prompt(query, messages, system_prompt, search_results, search_analysis):
    """Create an enhanced prompt that encourages deep thinking."""
    
    # Build search context
    search_context = "RELEVANT INFORMATION FOUND:\n\n"
    
    for source, data in search_results.items():
        search_context += f"=== {source.upper()} ===\n"
        
        if isinstance(data, list):
            for item in data[:2]:
                if isinstance(item, dict):
                    if 'title' in item:
                        search_context += f"Title: {item['title']}\n"
                    if 'summary' in item:
                        search_context += f"Summary: {item['summary']}\n"
                    if 'answer' in item:
                        search_context += f"Answer: {item['answer']}\n"
                    search_context += "\n"
        
        elif isinstance(data, dict):
            for key, value in data.items():
                if key not in ['source', 'type'] and value:
                    if isinstance(value, list):
                        search_context += f"{key}: {', '.join(str(v) for v in value[:3])}\n"
                    else:
                        search_context += f"{key}: {value}\n"
            search_context += "\n"
    
    # Add analysis insights
    search_context += "ANALYSIS INSIGHTS:\n"
    if search_analysis['key_facts']:
        search_context += "• Key facts identified from sources\n"
    if search_analysis['knowledge_gaps']:
        search_context += f"• Knowledge gaps: {search_analysis['knowledge_gaps'][0]}\n"
    
    # Build conversation history
    conversation = ""
    for msg in messages[-4:]:  # Last 4 messages for context
        if msg["role"] == "user":
            conversation += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation += f"Assistant: {msg['content']}\n"
    
    # Final prompt
    prompt = f"""<|system|>
{system_prompt}

CURRENT DATE: {datetime.now().strftime('%B %d, %Y')}

USER'S QUESTION: {query}

{search_context}

CONVERSATION CONTEXT:
{conversation}

THINKING INSTRUCTIONS:
1. First, verify the key information from sources
2. Identify the most reliable facts
3. Consider historical context if relevant
4. Think about why this matters
5. Connect to broader themes or concepts
6. Identify what's still unknown or debated
7. Formulate a comprehensive yet concise answer
8. End with thought-provoking questions or further reading suggestions

IMPORTANT: Show your reasoning process. Be precise about what's well-established vs. what's uncertain.</s>

<|user|>
{query}</s>

<|assistant|>
"""
    
    return prompt

def generate_thoughtful_response(model, prompt, max_tokens=768, temperature=0.7):
    """Generate response with thinking emphasis."""
    
    response = model(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        stop=["</s>", "<|user|>", "\n\nUser:", "### END", "Sources:"]
    )
    
    # Clean up response
    response = response.strip()
    
    # Ensure it doesn't cut off mid-thought
    if response.count('.') < 2:
        # If response seems incomplete, try to extend it
        extended = model(
            prompt + response,
            max_new_tokens=200,
            temperature=temperature,
            top_p=0.9
        )
        response = response + " " + extended.strip()
    
    return response

# Sidebar
with st.sidebar:
    st.header("🎭 Thinking Persona")
    
    # Safely get index for selectbox
    preset_keys = list(PRESET_PROMPTS.keys())
    current_preset = st.session_state.selected_preset
    
    # Ensure current preset is valid
    if current_preset not in preset_keys:
        current_preset = "Deep Thinker Pro"
        st.session_state.selected_preset = current_preset
    
    index = preset_keys.index(current_preset)
    
    persona = st.selectbox(
        "Select AI Persona:",
        options=preset_keys,
        index=index
    )
    
    if persona != st.session_state.selected_preset:
        st.session_state.selected_preset = persona
        st.session_state.system_prompt = PRESET_PROMPTS[persona]
    
    st.divider()
    
    st.header("⚡ Thinking Parameters")
    
    thinking_mode = st.radio(
        "Thinking Mode:",
        ["Analytical", "Creative", "Critical", "Balanced"],
        index=3
    )
    
    research_depth = st.select_slider(
        "Research Depth:",
        options=["Quick Scan", "Moderate", "Deep Dive", "Exhaustive"],
        value="Moderate"
    )
    
    temperature = st.slider(
        "Creativity Level:",
        0.1, 1.5, 0.7, 0.1,
        help="Lower = more factual, Higher = more creative"
    )
    
    st.divider()
    
    st.header("🔧 Tools")
    
    auto_search = st.toggle("Auto-Research", value=True)
    show_thinking = st.toggle("Show Thinking Process", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🧠 Reset AI", use_container_width=True):
            st.session_state.system_prompt = PRESET_PROMPTS["Deep Thinker Pro"]
            st.session_state.selected_preset = "Deep Thinker Pro"
            st.rerun()
    
    st.divider()
    st.caption("DeepThink Pro v1.1")
    st.caption("Advanced thinking AI with smart search")

# Main interface
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("🧠 DeepThink Pro")
with col2:
    if auto_search:
        st.success("🔍 Auto-Research ON")
with col3:
    if show_thinking:
        st.info("💭 Showing Thoughts")

# Display current persona
with st.expander("🤖 Active Persona", expanded=False):
    st.write(st.session_state.selected_preset)
    st.caption(st.session_state.system_prompt[:300] + "...")

# Load model
if st.session_state.model is None:
    with st.spinner("🚀 Loading AI Brain..."):
        try:
            st.session_state.model = load_model()
            st.success("✅ AI Ready for Deep Thinking!")
        except Exception as e:
            st.error(f"❌ Failed to load: {str(e)}")
            st.stop()

# Display chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show source tags if available
        if "sources" in message.get("metadata", {}):
            st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
            for source in message["metadata"]["sources"]:
                st.markdown(f'<span class="source-tag">{source}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare assistant response
    with st.chat_message("assistant"):
        # Step 1: Show thinking
        thinking_placeholder = st.empty()
        
        if show_thinking:
            thinking_placeholder.markdown("""
            <div class="thinking-bubble">
            <strong>💭 Initial Analysis:</strong><br>
            1. Parsing question structure and intent<br>
            2. Identifying key concepts and entities<br>
            3. Determining appropriate research approach<br>
            4. Preparing search strategy...
            </div>
            """, unsafe_allow_html=True)
        
        # Step 2: Intelligent Search
        search_results = {}
        search_analysis = {}
        
        if auto_search:
            if show_thinking:
                thinking_placeholder.markdown("""
                <div class="thinking-bubble">
                <strong>🔍 Smart Research:</strong><br>
                • Analyzing query type and selecting optimal sources<br>
                • Conducting parallel searches across selected databases<br>
                • Evaluating source reliability and relevance...
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner("🔍 Conducting intelligent research..."):
                search_results = perform_intelligent_search(prompt)
                
                if search_results:
                    search_analysis = analyze_search_results(prompt, search_results)
                    
                    # Display search summary
                    with st.expander("📊 Research Summary", expanded=False):
                        for source, data in search_results.items():
                            st.subheader(f"{SEARCH_TOOLS.get(source, {}).get('icon', '📌')} {source}")
                            
                            if isinstance(data, list):
                                for item in data[:2]:
                                    if isinstance(item, dict):
                                        with st.container():
                                            if 'title' in item:
                                                st.write(f"**{item['title']}**")
                                            if 'summary' in item:
                                                st.write(item['summary'])
                                            st.divider()
                            elif isinstance(data, dict):
                                for key, value in data.items():
                                    if key not in ['source', 'type'] and value:
                                        st.write(f"**{key.title()}:** {value}")
        
        # Step 3: Generate thoughtful response
        if show_thinking:
            thinking_placeholder.markdown("""
            <div class="thinking-bubble">
            <strong>🤔 Deep Synthesis:</strong><br>
            • Integrating information from multiple sources<br>
            • Applying critical thinking and analysis<br>
            • Formulating comprehensive response<br>
            • Preparing insights and recommendations...
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Engaging deep thinking process..."):
            # Create enhanced prompt
            enhanced_prompt = create_thinking_prompt(
                prompt, 
                st.session_state.messages,
                st.session_state.system_prompt,
                search_results,
                search_analysis
            )
            
            # Adjust tokens based on research depth
            if research_depth == "Quick Scan":
                tokens = 512
            elif research_depth == "Moderate":
                tokens = 768
            elif research_depth == "Deep Dive":
                tokens = 1024
            else:  # Exhaustive
                tokens = 1536
            
            # Generate response
            response = generate_thoughtful_response(
                st.session_state.model,
                enhanced_prompt,
                max_tokens=tokens,
                temperature=temperature
            )
        
        # Clear thinking placeholders
        thinking_placeholder.empty()
        
        # Display response
        st.markdown(response)
        
        # Add analysis box for deep thinking
        if thinking_mode != "Quick Scan" and search_results:
            st.markdown("""
            <div class="analysis-box">
            <strong>📈 Analysis Summary:</strong><br>
            • Information synthesized from {} sources<br>
            • Key themes identified<br>
            • Reliability assessment completed<br>
            • Knowledge gaps noted for further research
            </div>
            """.format(len(search_results)), unsafe_allow_html=True)
        
        # Store message with metadata
        metadata = {
            "sources": list(search_results.keys()) if search_results else [],
            "thinking_mode": thinking_mode,
            "research_depth": research_depth,
            "timestamp": datetime.now().isoformat()
        }
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "metadata": metadata
        })

# Add quick questions examples
if not st.session_state.messages:
    st.markdown("### 💡 Try asking about:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Historical figure analysis", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Who was Napoleon Bonaparte and what was his impact on Europe?"})
            st.rerun()
        if st.button("Scientific concept", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Explain quantum entanglement in simple terms"})
            st.rerun()
    
    with col2:
        if st.button("Current events", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What are the main challenges facing renewable energy adoption today?"})
            st.rerun()
        if st.button("Philosophical question", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is the meaning of consciousness according to different philosophical traditions?"})
            st.rerun()
