"""
Response generation using Groq LLM
"""

import re
import logging
from typing import List, Dict, Any
from groq import Groq

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate responses using Groq LLM"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key
            model: Model identifier
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        
        self.system_prompt = """You are an expert first aid assistant with access to authoritative medical sources. 

**Response Format**:

WARNING: Immediate Action
[Critical first steps - include "CALL 911/999 IMMEDIATELY" for life-threatening situations]

Step-by-Step Instructions
1. [First action]
2. [Second action]
3. [Continue with clear steps]

When to Seek Medical Help
- [Warning sign 1]
- [Warning sign 2]

What NOT to Do
- [Avoid 1]
- [Avoid 2]

Additional Notes
[Important context, warnings, or tips]


**Guidelines**:
- Use simple, clear language
- Be specific and actionable
- Always prioritize safety
- Cite sources used
- If information is limited, say so clearly
- For emergencies, emphasize calling 911/999"""
        
        logger.info(f"Response generator initialized with model: {model}")
    
    def clean_response_format(self, text: str) -> str:
        """
        Remove markdown artifacts and normalize formatting
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned text
        """
        if not text:
            return text

        # Remove markdown headings
        text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)

        # Remove bold/italic markup
        text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text, flags=re.DOTALL)

        # Remove inline code
        text = re.sub(r'`(.+?)`', r'\1', text, flags=re.DOTALL)

        # Normalize bullet characters
        text = re.sub(r'(?m)^[\s]*[-\*\u2022]\s+', '- ', text)

        # Collapse excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()
    
    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Generate response using Groq LLM
        
        Args:
            query: User query
            context_chunks: Retrieved relevant chunks
            conversation_history: Optional conversation history
            
        Returns:
            Generated response
        """
        # Build context from chunks
        context_text = "\n\n---\n\n".join([
            f"Source {i+1} ({chunk['metadata'].get('source', 'Unknown')}):\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks[:5])
        ])
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history[-4:])  # Last 2 exchanges
        
        # Add current query with context
        user_message = f"""Based on the following authoritative first aid information, answer this question:

Question: {query}

Relevant Information:
{context_text}

Provide clear, actionable first aid guidance following the response format."""

        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048
            )
            
            response = completion.choices[0].message.content
            return self.clean_response_format(response)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_fallback_response(self, query: str) -> str:
        """
        Generate fallback response when no relevant chunks found
        
        Args:
            query: User query
            
        Returns:
            Fallback response
        """
        # Detect context from keywords
        query_lower = query.lower()
        
        responses = {
            "nausea": (
                "Possible causes: food poisoning, viral infection, dehydration, motion sickness.\n\n"
                "Relief Steps:\n"
                "- Sit or lie down in a comfortable position.\n"
                "- Sip water slowly or try ginger tea.\n"
                "- Avoid solid foods until nausea passes.\n"
                "- Get fresh air if possible.\n\n"
                "When to Seek Medical Help:\n"
                "- Vomiting lasts more than 24 hours or you can't keep fluids down.\n"
                "- There's blood in vomit or severe stomach pain."
            ),
            "headache": (
                "Possible causes: tension headache, dehydration, migraine, stress, or heat.\n\n"
                "Relief Steps:\n"
                "- Rest in a quiet, dark room.\n"
                "- Drink water - dehydration can worsen headaches.\n"
                "- Apply a cold compress to your forehead.\n\n"
                "When to Seek Medical Help:\n"
                "- The headache is sudden and severe.\n"
                "- Vision changes, confusion, or vomiting occur."
            ),
            "dizziness": (
                "Possible causes: dehydration, low blood sugar, fatigue, or fainting onset.\n\n"
                "What to Do:\n"
                "- Sit or lie down immediately.\n"
                "- Drink water or an electrolyte solution.\n"
                "- Eat something light if you haven't eaten recently.\n\n"
                "When to Seek Medical Help:\n"
                "- Dizziness lasts long or occurs with chest pain or shortness of breath."
            ),
        }
        
        # Try to match a response
        for keyword, response in responses.items():
            if keyword in query_lower:
                return f"Based on your symptoms, this may indicate {keyword.title()}.\n\n{response}\n\nAdditional Notes:\nThis is general first aid guidance. If symptoms worsen or you're unsure, seek professional medical advice."
        
        # Generic fallback
        return (
            f"I couldn't find specific information for: \"{query}\".\n\n"
            "General First Aid Steps:\n"
            "1. Ensure safety and check responsiveness.\n"
            "2. Call emergency services if pain, bleeding, or confusion is severe.\n"
            "3. Provide rest, hydration, and reassurance.\n"
            "4. Monitor symptoms and avoid unnecessary movement.\n\n"
            "Additional Notes:\n"
            "If symptoms worsen or persist, consult a doctor immediately."
        )
