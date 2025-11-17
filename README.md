# 📧 AI-Powered Email Classification & Auto-Response System

This project intelligently classifies emails as **Spam / Not Spam**, analyzes **sentiment**, and generates an appropriate **safe or helpful reply** using **LangChain** and **Hugging Face models**.  
A clean **Streamlit UI** is included for smooth real-time testing.

---

## 🧩 Tech Stack

| Component | Description |
|------------|--------------|
| 🐍 **Python** | Core programming language |
| 🔗 **LangChain** | LLM orchestration & chain logic |
| 🤗 **Hugging Face Inference API** | Large Language Models for text generation |
| 🧱 **Pydantic** | Structured and validated output models |
| 🌐 **Streamlit** | Frontend UI for live email testing |

---

## ⚡ How It Works

1. Email is passed into the main LangChain pipeline.  
2. A **spam classifier chain** first determines whether the email is spam.  
3. If SPAM → system triggers a **warning-based spam response chain**.  
4. If NOT SPAM → email goes to **sentiment analysis** (positive/negative).  
5. Based on sentiment, system generates a **context-aware professional reply**.  
6. Streamlit UI displays the generated output in a modern card layout.

---

## 💡 Learning Highlights

- Built modular chain architecture using **RunnableBranch**, **Sequential**, and **Lambda** patterns.  
- Designed strict output formats using **Pydantic** for reliability.  
- Integrated Hugging Face models through **ChatHuggingFace** for structured generation.  
- Connected backend chains to a clean **Streamlit interface**.  
- Improved understanding of how complex LLM workflows operate end-to-end.

---

## 🎥 Demonstration Video

### **➡️ Demonstration video link is here:**  
🔗 *[LinkedIn Video Link — Add Your Link Here]*

---

## 🧪 Run It Yourself

Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
