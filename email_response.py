import streamlit as st 
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from pydantic import BaseModel , Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser , StrOutputParser
from langchain_core.runnables import RunnableBranch , RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    #repo_id="deepseek-ai/DeepSeek-R1",
    repo_id="deepseek-ai/DeepSeek-V3.1",
    
    task="text-generation",
    temperature=0.7

)

model = ChatHuggingFace(llm = llm)


#define category spam 

class category_spam(BaseModel):
    category_spam : str = Field(description="Give category of email in spam or not spam")

parser_spam = PydanticOutputParser(pydantic_object=category_spam)

prompt_spam = PromptTemplate(
    template="email - > {email} \n {formate_instruction}",
    input_variables=["email"],
    partial_variables={"formate_instruction":parser_spam.get_format_instructions()}
)

#define sentiment category

class Sentiment_category(BaseModel):
    sentiment : str = Field(description="Give category of email in positive or negative")

parser_sentiment = PydanticOutputParser(pydantic_object=Sentiment_category)

prompt_sentiment = PromptTemplate(
    template="email - > {email} \n {formate_instruction}",
    input_variables=["email"],
    partial_variables={"formate_instruction":parser_sentiment.get_format_instructions()}
)

#normal response 
class response(BaseModel):
    reply : str = Field(description="Give appropriate response on email ")

parser_response = PydanticOutputParser(pydantic_object=response)

prompt_response = PromptTemplate(
    template="email - > {email}  \n {formate_instruction}",
    input_variables=["email"],
    partial_variables={"formate_instruction":parser_response.get_format_instructions()}
)

#spam response 
class spam_response(BaseModel):
    reply : str = Field(description="A Warnning message !!!! ")

praser_spam_resp = PydanticOutputParser(pydantic_object=spam_response)

prompt_spam_resp = PromptTemplate(
    template=("The email is classified as SPAM.\n"
        "Generate a safe warning response.\n\n"
        "{format_instruction}"),
    partial_variables={"format_instruction":praser_spam_resp.get_format_instructions()}
)

    

#chains

spam_category_chain = prompt_spam | model | parser_spam

pos_neg_category_chain = prompt_sentiment | model | parser_sentiment

response_chain = prompt_response | model | parser_response

spam_response_chain = prompt_spam_resp | model | praser_spam_resp

chain = (
    {
        "email": lambda x: x["email"],
        "spam_result": lambda x: spam_category_chain.invoke({"email": x["email"]}),
    }
    |
    RunnableBranch(
        # Condition 1 → If spam
        (
            lambda x: x["spam_result"].category_spam.lower().strip() == "spam",
            spam_response_chain,
        ),

       (
            {
                "email": lambda x: x["email"],
                "sentiment": lambda x: pos_neg_category_chain.invoke({"email": x["email"]}).sentiment,
            }
            | response_chain
        )
    )
)

#streamlit UI

st.set_page_config(
    page_title="Smart Email Analyzer",
    page_icon="📨",
    layout="centered"
)

# MODERN CARD-BASED BLUE THEME
st.markdown("""
<style>

body {
    background: #e9eff9;
    font-family: 'Segoe UI', sans-serif;
}

/* TITLE */
.title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    color: #1a4fb8;
    margin-top: 20px;
}

/* SUBTITLE */
.subtitle {
    font-size: 15px;
    text-align: center;
    color: #4a4a4a;
    margin-bottom: 35px;
}

/* MODERN CARD */
.card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border-radius: 18px;
    padding: 30px;
    margin: auto;
    width: 90%;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

/* BUTTON */
.stButton>button {
    background: #1a4fb8;
    color: white;
    padding: 10px 25px;
    border-radius: 10px;
    border: none;
    transition: 0.25s;
    font-size: 15px;
    font-weight: 600;
}
.stButton>button:hover {
    background: #153f90;
}

/* RESPONSE CARDS */
.response-card {
    background: #eef4ff;
    border-left: 5px solid #1a4fb8;
    padding: 20px;
    border-radius: 12px;
    margin-top: 25px;
    color: #0f2f6e;
    font-size: 16px;
}

.spam-card {
    background: #ffeaea;
    border-left: 5px solid #d62828;
    padding: 20px;
    border-radius: 12px;
    margin-top: 25px;
    color: #7a1212;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="title">📨 Smart Email Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Spam Detection • Sentiment Analysis • Auto-Reply Generation</div>', unsafe_allow_html=True)

# MAIN CARD
st.markdown('<div class="card">', unsafe_allow_html=True)

email = st.text_area("✉️ Paste Email Content Here:", height=180)

if st.button("Analyze Email"):
    if not email.strip():
        st.warning("Please enter an email!")
    else:
        with st.spinner("Analyzing email..."):
            result = chain.invoke({"email": email})

        # Determine which card to show
        if "Warning" in result.reply or "spam" in result.reply.lower():
            st.markdown(
                f'<div class="spam-card"><b>🚨 SPAM DETECTED</b><br><br>{result.reply}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="response-card"><b>💬 Response</b><br><br>{result.reply}</div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)

