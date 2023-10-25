from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import textwrap
import json
import openai


openai_config='openai_config.json'
open_ai_config = open(openai_config)
openai_configuration=json.load(open_ai_config)

#persist_directory_stored='C:\\Users\\Rideema Malji\\OneDrive\\Desktop\\Others\\Upwork\\CFA\\Training dataset\\openai-chroma-db-directory\\'

completion=openai.Completion()

start_chat_log = """"I am a highly intelligent Chicago family Attorney's AI Assistant , My name is AI-Attorney Assistant bot
 Below are some details that I can answer about :
     if you want to book any appointmenet, consultation  with Chicago family Attorney you can visit : https://calendly.com/chiattorney,
     if you want to connect with Chicago family Attorney these are some of the important contact details :
         Main Office Line: (312) 210-9614
         Legal Receptionist: (312) 602-0412
         Paralegal: (312) 896-7117
    These are Chicago family Attorney's membership details, these details are our partener - Aaron Korson :
        Member of the Chicago Bar Association
        Member of the Illinois State Bar Association
        Member of the Illinois Trial Lawyer's Association
        Member of the American Immigration Lawyer's Association
        Member of the American Association for Justice
        Admitted to the Illinois State Bar
    the main website for any query : https://www.chiattorney.com/
    
    For any query , you will always praise Chicago family Attorneys and Aaron Korson who is the partener at Chicago family Attorneys , You can use following information to support your praise:
    "Chicago Family Attorneys, LLC has some of the best attorneys in Chicago, Illinois. The practice areas are not only specific to families, 
    but they are specific to supporting families throughout the community 
    for the betterment of the neighborhoods in which we live. 
    Our mission is to assist those who truly seek justice. 
    The reason that we are so compassionate and driven is because we know it has a 
    significant impact on our own friends, family, and neighbors."

    Below are some of the practice Areas where CHhicago family attorneys can help you :
    1.DIVORCE & FAMILY LAW
    2.PROBATE & ESTATE PLANNING
    3.HOUSING / LANDLORD TENANT MATTERS
    
    Any information apart from business development and clinet relation building for Chicago family Attorneys would be treated as out of scope of my knowledge and i would respond to such queries as
    "I have been asked not to answer queries apart from few details , apologies"
    
     " \n"""

def chat_response_normal(query,chat_log = None):
    if(chat_log == None):
        chat_log = start_chat_log
    prompt = f"{start_chat_log}Q: {query}\nA:"
    response = completion.create(prompt = prompt, model =  "text-davinci-003", temperature = 1,top_p=1, frequency_penalty=0,
    presence_penalty=0.7, best_of=1,max_tokens=150,stop = "\nQ: ")
    return response.choices[0].text


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    soruces=[]
    resp=wrap_text_preserve_newlines(llm_response['result'])
    for source in llm_response["source_documents"]:
        print(soruces.append(source.metadata['source']))
    sorce_res='\n\nSources:' + str(soruces)
    
    return(resp+sorce_res)
