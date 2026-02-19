import os
from google import genai
from dotenv import load_dotenv  # 1. Import the library

# 2. Load the variables from the .env file
load_dotenv()

# 3. Get the key from the environment
api_key = os.getenv("GEMINI_API_KEY")

# 4. Create client
client = genai.Client(api_key=api_key)


def convert_to_sentence(word_list):
#prompt rules
    prompt = f"""
    You are a professional Malaysian BIM sign language interpreter.

    Rules:
    - Input is a list of keywords from sign language.
    - Rearrange into a natural Bahasa Melayu sentence.
    - Add missing pronouns or grammar if needed.
    - Keep meaning accurate.
    - Do NOT explain anything.
    - Return only one sentence.

    Input: {word_list}
    Output:
    """

    #Create client to call gemini api
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", 
        contents=prompt
    )

    return response.text


# ===== TEST =====

word_buffer = ["Sudah","Nasi", "Makan"]   #<--- This is the where the translated word is inputed

sentence = convert_to_sentence(word_buffer)

print(sentence)
