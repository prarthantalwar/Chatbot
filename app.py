from flask import Flask, render_template, request
import openai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter

# from langchain_community.embeddings import (
#     OpenAIEmbeddings,
#     HuggingFaceInstructEmbeddings,
# )
from langchain_openai import OpenAIEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain_community.vectorstores import FAISS

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub

# from htmlTemplates import css, bot_template, user_template

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# print(openai.api_key)

# Set up Flask app
app = Flask(__name__)


def get_pdf_text(file_path):
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = conversation.invoke({"question": user_question})
    chat_history = response["chat_history"]

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            temp = "user_template"
            mess = message.content
        else:
            temp = "bot_template"
            mess = message.content
    return temp, mess


# Define the home page route
@app.route("/")
def home():
    return render_template("index.html")


# Define the Chatbot route
@app.route("/chatbot", methods=["POST"])
def chatbot():
    # Get the message input from the user
    user_input = request.form["message"]
    # Use the OpenAI API to generate a response
    prompt = f"User: {user_input}\nChatbot:"
    chat_history = []

    temp, mess = handle_userinput(user_input)

    # response = openai.Completion.create(
    #     engine="davinci-002",
    #     prompt=prompt,
    #     temperature=0.5,
    #     max_tokens=60,
    #     top_p=1,
    #     frequency_penalty=0,
    #     stop=["\nUser: ", "\nChatbot: "],
    # )

    # # Extract the response text from the OpenAI API result
    # bot_response = response.choices[0].text.strip()
    # # Add the user input and bot response to the chat history
    # chat_history.append(f"User: {user_input}\nChatbot: {bot_response}")

    # Render the Chatbot template with the response text
    return render_template(
        "chatbot.html",
        user_input=user_input,
        bot_response=mess,
    )


raw_text = get_pdf_text("essay.pdf")
text_chunks = get_text_chunks(raw_text)
vectorstore = get_vectorstore(text_chunks)
conversation = get_conversation_chain(vectorstore)

# print("\n" * 200)
# for i in text_chunks:
#     print(i)
# print(text_chunks)


# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
