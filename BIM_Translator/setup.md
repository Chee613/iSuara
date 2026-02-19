## 🔧 Setup Guide – BIM Sign-to-Text Translator

Follow the steps below to set up the project locally.

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd BIM_Translator
```

---

### 2️⃣ Create a Virtual Environment

It is recommended to use a virtual environment to prevent library conflicts.

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows (PowerShell):**
```bash
venv\Scripts\activate
```

If execution is blocked, run:
```bash
Set-ExecutionPolicy RemoteSigned
```

---

### 3️⃣ Install Dependencies

```bash
pip install google-genai
pip install python-dotenv
```

(Install additional dependencies such as MediaPipe, Scikit-learn, or TensorFlow if required.)

---

### 4️⃣ Configure Environment Variables

Create a `.env` file in the root directory:

```
GOOGLE_API_KEY=your_api_key_here
```

Obtain your API key from:
https://aistudio.google.com/app/apikey

⚠️ Do not share your API key publicly.

---

### 5️⃣ Run the Translator

```bash
python translator.py
```

If configured correctly, the system will convert the detected sign keywords into a grammatically correct Bahasa Melayu sentence using Gemini.

---

## 🧠 Project Architecture Overview

1. **Step 1 – Sign Recognition**  
   MediaPipe extracts hand landmarks and a trained classifier predicts individual sign words.

2. **Step 2 – Word Buffer Logic**  
   Words are stored in a buffer with stability filtering and pause detection.

3. **Step 3 – Sentence Formation**  
   Gemini (gemini-2.5-flash-lite) restructures keywords into natural Bahasa Melayu sentences.

---

## 🚀 Recommended Best Practices

- Always use a virtual environment.
- Store API keys in `.env` files.
- Add `.env` and `venv/` to `.gitignore`.
- Avoid hardcoding credentials.

---

This setup ensures a clean, modular, and scalable pipeline for real-time BIM Sign-to-Text translation.
