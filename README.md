# 📧 AI-Powered Email Classification & Auto-Response System

This project intelligently classifies emails as **Spam / Not Spam**, analyzes **sentiment**, and generates an appropriate **safe or helpful reply** using LangChain and Hugging Face models.  
A clean **Streamlit UI** is included for smooth real-time testing.

---

## 🎥 Demonstration Video

### **➡️ Demonstration video link is here:**  
🔗 *[LinkedIn Video Link — Add Your Link Here]*

---

## Test Result 
<p align="center">
  <img src="https://github.com/Parth-Mahadik-1/langchain-chains/blob/main/screenshots/chain%20ss/simple.png" width="550" />

</p>

## 🧩 Tech Stack

| Component | Description |
|------------|--------------|
| 🐍 **Python** | Core programming language |
| 🔗 **LangChain** | LLM orchestration & chain logic |
| 🤗 **Hugging Face Inference API** | Models for text generation & classification |
| 🧱 **Pydantic** | Structured output validation |
| 🌐 **Streamlit** | Frontend UI for real-time interaction |

---

## ⚡ How It Works

1. Email is passed into the main LangChain pipeline.  
2. A **spam classifier chain** evaluates if the email is spam.  
3. If SPAM → triggers a **warning-based spam response chain**.  
4. If NOT SPAM → email goes through **sentiment analysis**.  
5. Based on sentiment, a **professional context-aware reply** is generated.  
6. Streamlit UI displays all outputs in a **modern card-based layout**.

---

## 💡 Learning Highlights

- Gained hands-on experience with **RunnableBranch**, conditional chains, and modular LLM logic.  
- Understood **Pydantic’s structured outputs** for predictable responses.  
- Integrated Hugging Face models using **ChatHuggingFace**.  
- Developed a clean **Streamlit frontend** to interact with backend chains.  
- Built a fully functional **end-to-end LLM workflow**.

---

## 🧪 Run It Yourself

Clone the repository:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
