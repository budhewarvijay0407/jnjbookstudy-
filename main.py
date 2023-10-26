#Importing all the necessary language 
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from streamlit_lottie import st_lottie
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import openai
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import PyPDF2
import datetime as dt
global qa


qa=None
start_text=''

def get_embeddings(total_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    texts = text_splitter.create_documents(total_data)
    db = Chroma.from_documents(texts, embeddings,persist_directory='local_db')
    db.persist()
    return(db)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

##Styling part - this css file beatify the look of our Streamlit - Many people dont know about this feature 
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style/style.css")


#Conifguration part where we are reading the necessary variables from outside - This practice makes the code flexible and deployable 
openai_config='openai_config.json'  #give standard path
open_ai_config = open(openai_config)
openai_configuration=json.load(open_ai_config)
os.environ['OPENAI_API_KEY']=openai_configuration['key']
openai.api_key=openai_configuration['key']
embeddings = OpenAIEmbeddings()

with st.sidebar:
    st.sidebar.image('j&jlogo.jpeg')
    lotti_sidebar=load_lottieurl('https://lottie.host/75d77fdd-d088-422c-a94f-505ee47fd5ee/6KEpCAcDQD.json')
    st_lottie(lotti_sidebar,reverse=True,height=300,  width=300,speed=1,  loop=True,quality='high')
    st.title("BookStudyPrep")
    
    # st.markdown('''This application showcases the capabilities of AI using OpenAI's LLMs
    # ''')
    date=dt.datetime.now()
    st.markdown(f"**Session Started/Refreshed @ {date}**")
    

with st.container():
    
    if 'chroma_db' not in st.session_state:
        st.session_state['chroma_db']=''
    
    if 'start_text' not in st.session_state:
        st.session_state['start_text']=''
        
    if 'generated_qa' not in st.session_state:
        #print('Inside')
        st.session_state['generated_qa'] = []
    
    if 'past_qa' not in st.session_state:
        st.session_state['past_qa'] = []

    def get_text_qa():    
        input_text_qa = st.text_input("Your Question: ", "", key="input_text_qa_1")
        return(input_text_qa)
    

    
    def resp_qa():
        with response_container_qa:
            if user_input_qa:
                print('User input is :',user_input_qa)
                if 'okay' in user_input_qa.lower():
                    response_qa = {'query':'','result':'Can i help you with anything else?'}
                if 'thank' in user_input_qa.lower():
                    response_qa = {'query':'','result':'You are welcome ,Can i help you with anything else?'}
                else:
                    response_qa = qa(user_input_qa)
                print('The response is ',response_qa)
                st.session_state.past_qa.append(user_input_qa)
                st.session_state.generated_qa.append(response_qa['result'])
                
            if st.session_state['generated_qa']:
                for i in range(len(st.session_state['generated_qa'])):
                    message(st.session_state['past_qa'][i], is_user=True, key=str(i) + '__user')
                    message(st.session_state['generated_qa'][i], key=str(i)+'g')
                with reference_document:
                    if response_qa != None:
                        try:
                            st.write('The reference for the lastest question asked is:',response_qa['source_documents'][1])
                        except:
                            st.write('Empty reference')
                            
    st.write("*The Chat interface and reference corpus interface would appear once you upload the file*")
    
    uploaded_file = st.file_uploader('Upload pdf', type='pdf', accept_multiple_files=False, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    response_container_qa = st.container()
    colored_header(label='', description='', color_name='yellow-30')
    
    print('upload file first:',uploaded_file)
    if "uploaded_file_s" not in st.session_state:
        st.session_state.uploaded_file_s = False

    if uploaded_file or st.session_state.uploaded_file_s:
        st.session_state.uploaded_file_s = True
        if uploaded_file is not None:
            with st.spinner('processing the document'):
                total_data=[]
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                if len(pdf_reader.pages)>20:
                    len_read=20
                else:
                    len_read=len(pdf_reader.pages)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    total_data.append(text)
                if st.session_state['start_text']!=total_data:
                    st.session_state['start_text']=total_data
                    chroma_db=get_embeddings(total_data)
                    st.session_state['chroma_db']=chroma_db
                 
                vectordb_openai = Chroma(persist_directory='local_db', embedding_function=OpenAIEmbeddings())
                retriever_openai = vectordb_openai.as_retriever(search_kwargs={"k": 2})
                qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name="text-davinci-003",max_tokens=200), chain_type="stuff", retriever=retriever_openai,return_source_documents=True)
            #st.success('')
        input_container_qa = st.container()
        reference_document = st.container()
        with input_container_qa:
            user_input_qa = get_text_qa()
            print('I got the input text as :',user_input_qa)
            resp_qa()
