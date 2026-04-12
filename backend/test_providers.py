import asyncio
import os
import sys
from dotenv import load_dotenv

async def main():
    load_dotenv("/Users/mohammedasrarali/Desktop/Deep_research/.env")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("Error: Could not import langchain libraries. Make sure to use the virtual environment.")
        sys.exit(1)

    print("Testing Providers...")
    
    # Gemini
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", max_retries=0)
        res = await llm.ainvoke("ping")
        print("✅ Gemini: WORKING (Quota has reset!)")
    except Exception as e:
        print(f"❌ Gemini: {e}")

    # Groq Primary
    try:
        llm = ChatOpenAI(model="llama-3.3-70b-versatile", base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"), max_retries=0)
        res = await llm.ainvoke("ping")
        print("✅ Groq Primary (llama-3.3-70b): WORKING")
    except Exception as e:
        print(f"❌ Groq Primary: {e}")
        
    # Groq Secondary
    try:
        llm = ChatOpenAI(model="qwen/qwen3-32b", base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"), max_retries=0)
        res = await llm.ainvoke("ping")
        print("✅ Groq Secondary (qwen3-32b): WORKING")
    except Exception as e:
        print(f"❌ Groq Secondary: {e}")

    # Groq Tertiary
    try:
        llm = ChatOpenAI(model="meta-llama/llama-4-scout-17b-16e-instruct", base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"), max_retries=0)
        res = await llm.ainvoke("ping")
        print("✅ Groq Tertiary (llama-4-scout): WORKING")
    except Exception as e:
        print(f"❌ Groq Tertiary: {e}")

if __name__ == "__main__":
    asyncio.run(main())
