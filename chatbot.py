# This chatbot script is adapted from the example provided by @Aakriti E-Learning Academy on Udemy

import streamlit as st
#comment all following for streamlit deploy
#from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI

#Pass your key here
OPENAI_API_KEY = "OPENAI_API_KEY"

custom_css = '''
<style>
h2 {
    text-align: center;
}
</style>
'''

# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

st.header("Nan's Chatbot")
st.write("Welcome to my chatbot, powered by the OpenAI API and deployed using Streamlit! This chatbot allows you to upload your documents, train it specifically on your content, and ask related questions.")
# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)
#Upload PDF files
with  st.sidebar:
    st.title("Hi, there!")
    st.write("I'm Nan's first chatbot")
    file = st.file_uploader(" Upload function is currently unavailable", type="pdf")
    #file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")


#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators= ["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)
    #get user input
    user_question = st.text_input("After upload your file, type your question here")

    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        #define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-4.0"
        )

        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
else:# for demo purpose
    user_question = st.text_input("Let's chat! You can talk to me here.")
    if user_question:
        response = "This chatbot uses the OpenAI API, and currently, I don't have sufficient credits to provide a full experience. However, you can check out the script and code on my [GitHub](https://github.com/xiaonanshi/chatbot)."
        st.write(response)

