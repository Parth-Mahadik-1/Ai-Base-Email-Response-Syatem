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

#Testing 

#testing spam chain
print("----- TEST SPAM CHAIN -----")
test_email_1 = "Congratulations! You won a free iPhone. Click here."

result = spam_category_chain.invoke({"email": test_email_1})
print(result)

#testing sentiment chain

print("----- TEST SENTIMENT CHAIN -----")
test_email_2 = "I am really happy with your service."

result = pos_neg_category_chain.invoke({"email": test_email_2})
print(result)

# testing spam respnse chain 

print("----- TEST SPAM RESPONSE CHAIN -----")
result = spam_response_chain.invoke({"":""})
print(result)

#testing sentiment response chain
print("----- TEST NORMAL RESPONSE CHAIN -----")
email = "Thank you for your help."
sentiment = "positive"