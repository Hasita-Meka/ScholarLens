"""
AI integration for ScholarLens
Supports Google AI Studio (Gemini) and OpenAI APIs
"""

import os
import re
import json
import time
from typing import List, Dict, Optional
from openai import OpenAI

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = None
provider = None
model_name = None

if GOOGLE_API_KEY:
    try:
        client = OpenAI(
            api_key=GOOGLE_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        provider = "google"
        model_name = "gemini-2.0-flash"
        print("Using Google AI Studio (Gemini) API")
    except Exception as e:
        print(f"Failed to initialize Google AI client: {e}")
        client = None
elif OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        provider = "openai"
        model_name = "gpt-4o-mini"
        print("Using OpenAI API")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        client = None


def is_available() -> bool:
    """Check if AI API is available"""
    return client is not None


def get_provider_info() -> Dict:
    """Get information about the current AI provider"""
    if not is_available():
        return {"available": False, "provider": None, "model": None}
    return {
        "available": True,
        "provider": provider,
        "model": model_name
    }


def _safe_api_call(func, *args, max_retries=2, **kwargs):
    """Wrapper for safe API calls with retry logic"""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            raise e


def _extract_json(text: str) -> str:
    """
    Extract JSON from a response that may contain extra text.
    Uses json.JSONDecoder to find the first valid JSON object/array.
    """
    text = text.strip()
    
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    for i, char in enumerate(text):
        if char in '[{':
            try:
                decoder = json.JSONDecoder()
                result, _ = decoder.raw_decode(text[i:])
                return json.dumps(result)
            except json.JSONDecodeError:
                continue
    
    return text


def _make_completion(messages: List[Dict], max_tokens: int = 1000, json_mode: bool = False) -> str:
    """Make a chat completion request with provider-specific handling"""
    if not is_available():
        raise Exception("AI API not configured")
    
    kwargs = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens
    }
    
    if json_mode and provider == "openai":
        kwargs["response_format"] = {"type": "json_object"}
    
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def generate_summary(text: str, audience: str = "expert") -> str:
    """
    Generate a summary tailored to a specific audience
    audience: 'expert', 'student', or 'policymaker'
    """
    if not is_available():
        return "AI API not configured. Please add GOOGLE_API_KEY or OPENAI_API_KEY to use AI features."
    
    audience_prompts = {
        "expert": """You are a research expert. Provide a technical summary of this research paper that:
- Highlights key methodological contributions
- Identifies novel techniques and their technical details
- Discusses experimental results with specific metrics
- Notes limitations and future research directions
Keep the summary concise but technically rigorous.""",
        
        "student": """You are an educational assistant helping students understand research. Provide a summary that:
- Explains complex concepts using simple language and analogies
- Breaks down the main research question and why it matters
- Describes the approach step by step
- Highlights key findings in accessible terms
- Suggests related topics to learn more about
Use clear, engaging language suitable for undergraduate students.""",
        
        "policymaker": """You are a policy advisor. Provide a summary focused on:
- Real-world applications and societal impact
- Potential benefits and risks
- Ethical considerations and concerns
- Policy implications and recommendations
- Timeline for practical deployment
Keep technical jargon minimal and focus on actionable insights."""
    }
    
    system_prompt = audience_prompts.get(audience, audience_prompts["expert"])
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize this research paper:\n\n{text[:15000]}"}
        ]
        return _make_completion(messages, max_tokens=1500)
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def answer_question(question: str, context: List[Dict], paper_title: str = "") -> Dict:
    """
    Answer a research question using RAG with source citations
    context: List of relevant text chunks with metadata
    """
    if not is_available():
        return {
            "answer": "AI API not configured. Please add GOOGLE_API_KEY or OPENAI_API_KEY to use AI features.",
            "sources": []
        }
    
    context_text = "\n\n".join([
        f"[Source {i+1}] (Paper: {c.get('paper_title', 'Unknown')}, Section: {c.get('section', 'Unknown')}):\n{c['content']}"
        for i, c in enumerate(context[:5])
    ])
    
    system_prompt = """You are a research assistant helping answer questions about scientific papers.

Rules:
1. Answer based ONLY on the provided context
2. Cite your sources using [Source N] format
3. If the context doesn't contain enough information, say so
4. Be precise and accurate
5. Provide specific quotes or data when relevant"""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Context from research papers:

{context_text}

Question: {question}

Please provide a detailed answer with citations."""}
        ]
        
        answer = _make_completion(messages, max_tokens=1000)
        
        sources = []
        for i, c in enumerate(context[:5]):
            if f"[Source {i+1}]" in answer or i < 2:
                sources.append({
                    "index": i + 1,
                    "paper_title": c.get('paper_title', 'Unknown'),
                    "section": c.get('section', 'Unknown'),
                    "content": c['content'][:300] + "..."
                })
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"Error generating answer: {str(e)}",
            "sources": []
        }


def generate_flashcards(text: str, num_cards: int = 5) -> List[Dict]:
    """
    Generate flashcards from paper content
    """
    if not is_available():
        return []
    
    try:
        system_content = f"""Generate exactly {num_cards} educational flashcards from this research paper.

Format your response as a JSON array with this exact structure:
[
    {{"question": "...", "answer": "...", "difficulty": "easy"}},
    {{"question": "...", "answer": "...", "difficulty": "medium"}},
    {{"question": "...", "answer": "...", "difficulty": "hard"}}
]

Focus on:
- Key concepts and definitions
- Important methods and their purposes
- Main findings and their significance
- Technical terms and their meanings

IMPORTANT: Return ONLY the JSON array, no other text."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Generate flashcards from:\n\n{text[:10000]}"}
        ]
        
        response_text = _make_completion(messages, max_tokens=1500, json_mode=True)
        json_text = _extract_json(response_text)
        
        result = json.loads(json_text)
        if isinstance(result, dict) and 'flashcards' in result:
            return result['flashcards']
        elif isinstance(result, list):
            return result
        else:
            return []
    except Exception as e:
        print(f"Error generating flashcards: {e}")
        return []


def generate_quiz(text: str, num_questions: int = 5) -> List[Dict]:
    """
    Generate quiz questions from paper content
    """
    if not is_available():
        return []
    
    try:
        system_content = f"""Generate exactly {num_questions} multiple-choice quiz questions from this research paper.

Format your response as JSON with this exact structure:
{{
    "questions": [
        {{
            "question": "What is the main contribution of this paper?",
            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
            "correct_answer": "A",
            "explanation": "Brief explanation of why this is correct"
        }}
    ]
}}

Questions should test understanding of:
- Core concepts and methods
- Research findings
- Technical details
- Practical applications

IMPORTANT: Return ONLY the JSON object, no other text."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Generate quiz questions from:\n\n{text[:10000]}"}
        ]
        
        response_text = _make_completion(messages, max_tokens=2000, json_mode=True)
        json_text = _extract_json(response_text)
        
        result = json.loads(json_text)
        return result.get('questions', [])
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return []


def generate_policy_brief(text: str, paper_title: str = "") -> str:
    """
    Generate a policy brief from paper content
    """
    if not is_available():
        return "AI API not configured. Please add GOOGLE_API_KEY or OPENAI_API_KEY to use AI features."
    
    try:
        system_content = """You are a policy analyst creating a brief for policymakers.

Structure the policy brief as follows:

## Executive Summary
Brief overview of the research and its significance

## Key Findings
Bullet points of main discoveries

## Applications
Real-world use cases and potential implementations

## Risks and Concerns
Potential negative impacts, ethical issues, and limitations

## Recommendations
Actionable policy suggestions

## Timeline
Expected development and deployment timeline

Keep language accessible to non-technical readers."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Create a policy brief for this research paper:\n\nTitle: {paper_title}\n\n{text[:12000]}"}
        ]
        
        return _make_completion(messages, max_tokens=2000)
    except Exception as e:
        return f"Error generating policy brief: {str(e)}"


def generate_analogy(concept: str, context: str = "") -> str:
    """
    Generate a cross-domain analogy for a research concept
    """
    if not is_available():
        return "AI API not configured."
    
    try:
        system_content = """You are an expert at creating intuitive analogies for complex technical concepts.

Rules:
1. Create an analogy using everyday experiences
2. The analogy should be accurate and capture the key essence
3. Explain how the analogy maps to the technical concept
4. Keep it concise but clear

Example: "Attention in NLP is like triaging patients in an ER - the most critical cases get immediate focus while others wait."
"""
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Create an intuitive analogy for this concept:\n\nConcept: {concept}\n\nContext: {context[:500] if context else 'No additional context'}"}
        ]
        
        return _make_completion(messages, max_tokens=300)
    except Exception as e:
        return f"Error generating analogy: {str(e)}"


def extract_key_insights(text: str) -> List[str]:
    """
    Extract key insights from paper text
    """
    if not is_available():
        return ["AI API not configured."]
    
    try:
        system_content = """Extract 5-7 key insights from this research paper.
                
Format as JSON:
{"insights": ["insight 1", "insight 2", ...]}

Each insight should be:
- A complete, standalone statement
- Focused on findings, methods, or implications
- Clear and concise

IMPORTANT: Return ONLY the JSON object, no other text."""

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text[:10000]}
        ]
        
        response_text = _make_completion(messages, max_tokens=800, json_mode=True)
        json_text = _extract_json(response_text)
        
        result = json.loads(json_text)
        return result.get('insights', [])
    except Exception as e:
        return [f"Error extracting insights: {str(e)}"]
