from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import time
import requests
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings, CohereEmbeddings
from typing import List
from sentence_transformers import SentenceTransformer
import logging
import json
from data import example_doc_feedback as ex
from data import parser_output as parser
import ast
import re

ASSISTANT_ID = 'asst_VSDOllgBf0GkI50zVnBJdJ5N'
# RETRIEVER_ID = 'asst_xMGkLqbVNc1EYAg14AFxpEdr'
NGROK_LINK = 'https://adb1-34-32-253-133.ngrok.io'

client = OpenAI()
thread = client.beta.threads.create()
# ret_thread = client.beta.threads.create()

RETRIEVER_INITIALIZED = False

class QueryHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.generated_queries = ""

    def emit(self, record):
        if record.levelname == 'INFO' and 'Generated queries' in record.msg:
            # Extract the text between '[' and ']'
            match = re.search(r'\[([^]]+)\]', record.msg)
            if match:
                queries = match.group(1)
                self.generated_queries = ""
                try:
                    queries_list = ast.literal_eval(queries)
                    for query in queries_list:
                        self.generated_queries += query + '\n'
                except (ValueError, SyntaxError) as e:
                    # Handle the exception as needed
                    print(f"Error evaluating queries: {e}")
                
# Configure logging
logging.basicConfig(filename='data/app.log', filemode='w', level=logging.WARNING)
logger = logging.getLogger("langchain.retrievers.multi_query")
logger.setLevel(logging.INFO)

# Add the custom handler to the logger
query_handler = QueryHandler()
logger.addHandler(query_handler)


def generate_document(api_doc_path: str, api_example_path: str, api_name: str):
    with open(api_doc_path, 'r') as f:
        data = json.load(f)
    
    doc_format = ''
    for i in data['ToolList']:
        if i['API Name'] == api_name:
            doc_format += f"##API Name: {i['API Name']} \n"
            doc_format += f"###Description: {i['API Description']}\n\n"
            doc_format += f'###Arguments: \n\n'
            for j in i['API arguments']:
                doc_format += f"API Argument: {j['Argument Name']}\n"
                doc_format += f"Argument Description: {j['Argument Description']}\n"
                doc_format += f"Return Type: {j['Argument Type']}\n"
                # doc_format += f"Value Examples: {j['Argument Value Examples']}\n"
                
    # with open(api_example_path, 'r') as f:
    #     ex_data = json.load(f)
    
    # ex_format = '\n\nExamples:\n'
    # for query in ex_data:
    #     if api_name in query['Output']:
    #         ex_format += f"###Query: {query['Query']}\n"
    #         ex_format += f"###Output: {query['Output']}\n"
    # return doc_format + ex_format
    
    return doc_format

def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def process_query(query: str) -> str:
    # client = OpenAI()
    sys_prompt = 'You are a friendly chatbot. The user will either enquire or will have a natural conversation. Each query would be resolved by a different pipeline in the backend which will generate a sequence of API calls in json format. Your goal is to *just* generate brief text (upto 35 words) that precedes the actual answer. Do not make any assumptions but act as if you are looking into it. You do not have to resolve any query so just send only one message in reply.'
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}
        ]
    )
    return completion.choices[0].message.content

    # message = client.beta.threads.messages.create(
    #     thread_id=thread.id,
    #     role="user",
    #     content=query
    # )
    # run = client.beta.threads.runs.create(
    #     thread_id=thread.id,
    #     assistant_id=ASSISTANT_ID,
    # )
    
    # wait_on_run(run, thread)
    
    # messages = client.beta.threads.messages.list(
    #     thread_id=thread.id,
    #     order='desc',
    # )
    
    # print(messages)
    # idx, ite = -1, 0
    # for i, msg in enumerate(messages):
    #     if msg.content[0].text.value == query:
    #         idx += i
    
    # for msg in messages:
    #     if ite == idx:
    #         return msg.content[0].text.value
    #     ite += 1


def retriever(query: str):
    if not RETRIEVER_INITIALIZED:
        class LineList(BaseModel):
            lines: List[str] = Field(description="Lines of text")

        class LineListOutputParser(PydanticOutputParser):
            def __init__(self) -> None:
                super().__init__(pydantic_object=LineList)

            def parse(self, text: str) -> LineList:
                lines = text.strip().split("\n")
                return LineList(lines=lines)
        api_list = []
        with open('./data/api_documentation.json', 'r') as f:
            data = json.load(f)
        for i in data['ToolList']:
            api_list.append(i['API Name'])
            
        doc = []
        meta = []
        id = []
        for itr, i in enumerate(api_list):
            api = {}
            doc.append(generate_document('./data/api_documentation.json', './data/examples.json', i))
            api["API"] = i
            meta.append(api)
            id.append(f"ID{itr}")

        output_parser = LineListOutputParser()
        
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            # template = "Repeat the word apple two times, like, \napple\napple",
            template="""You are a instructor your job is to break a query into smaller parts and provide it to worker. Given a conversation utterance by a user, ignore all the non-query part and try to break the main query into smaller steps. Don't include multiple steps, just whatever the query is trying to address. Output only the sub queries step by step and nothing else.
            Original question: {question}""",
        )
        llm = ChatOpenAI(temperature=0)
        llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)
        
        client_hf = chromadb.PersistentClient(path="./hf_db")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")
        collection_hf = client_hf.get_or_create_collection(name="hf_check_1", embedding_function = sentence_transformer_ef)
        collection_hf.add(
            documents=doc,
            metadatas=meta,
            ids=id
        )
        
        embeddings_hf = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

        vectorstore_hf = Chroma(
            collection_name="hf_check_1",
            embedding_function=embeddings_hf,
            persist_directory = "./hf_db"
        )
        retriever_from_llm = MultiQueryRetriever(
            retriever=vectorstore_hf.as_retriever(), llm_chain=llm_chain, parser_key="lines"
        )

    unique_docs_hf = retriever_from_llm.get_relevant_documents(
        query=query
    )
    return unique_docs_hf, query_handler.generated_queries


def generate_output(llm_in):
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": 'as'},
            {"role": "user", "content": llm_in}
        ],
        temperature=0.2,
        max_tokens = 250,
    )
    return completion.choices[0].message.content


def check_ans(query):
    payload = {"query": query}
    req = requests.post(
        NGROK_LINK+'/predict/', json=payload)
    return req.json()['text']


def feedback_part1(documentation_and_examples, input_query, model_output, temperature = 0.2, max_tokens = 250):

    updated_prompt = """You are an expert at analyzing API call sequences. Given an API call sequence and the input query, your job is to explain the task being performed by the API calls in small steps and determine whether the given query is answerable or unanswerable. Do not explain the API call, just output what it is doing. Output the small steps in points and nothing else. Keep the small steps as short and precise as possible. In order for the query to be classified as answerable, there should exist a correct sequence of API calls from the given set of API tools capable of solving the given query. If the query is unanswerable, do not break the API sequence into small steps. Just output a single word Answerable/Unanswerable. Look closely into the API tools' allowed arguments before deciding, for example: sometimes queries such as extract the 'top 5' might be unanswerable since the given API tools might not have any argument which can filter out a specified set of results, and using things like 'slice' would be wrong as it is not a valid API argument! Here is the API documentation along with some examples: """ + documentation_and_examples + """\n Here is the input query: """ + input_query + """\n here is the API call sequence: """ + model_output + """\n Keep your output as concise as possible and to the point."""

    feedback_response1 = client.chat.completions.create(
    model="gpt-4",
    messages = [{"role": "user", "content" : updated_prompt}],
    temperature=temperature,
    max_tokens = max_tokens,
    )
    return feedback_response1.choices[0].message.content


def feedback_part2(input_query, generated_query, temperature = 0.2, max_tokens = 150):
    feedback_prompt2 = """I am training an agent to generate the output based on an input query. Based on the output generated by the agent, I have written a generated query. You have to analyze how the generated query is different from the input query. If the generated query mentions unanswerable, just return "Unanswerable" do not output anything else. Otherwise, based on this, give feedback to the output-generating agent about how the output written by it should be modified to satisfy the input query. Keep the feedback precise and to the point. Do not mention the generated query in the feedback. Output only the feedback and nothing else.\n""" + input_query + """\n Here is the generated query: \n""" + generated_query
    
    feedback_response2 = client.chat.completions.create(
    model="gpt-4",
    messages = [{"role": "user", "content" : feedback_prompt2}],
    temperature=temperature,
    max_tokens = max_tokens,
    )
    
    return feedback_response2.choices[0].message.content


def get_feedback(input_query, model_output, documentation_and_examples):
    
    feedback_output1 = feedback_part1(documentation_and_examples, input_query, model_output)
    # print(feedback_output1)
    
    final_feedback = feedback_part2(input_query, feedback_output1)
    
    return final_feedback


def pipeline(query: str):
    start_time = time.time()

    model_in = ex.llm_input + '\n\n'
    retrieved_docs, sub_queries = retriever(query)
    
    retrieval_time = time.time() - start_time
    print(f"Retrieval Time: {retrieval_time} seconds")

    doc_text = ''
    for doc in retrieved_docs:
        doc_text += doc.page_content + '\n\n'

    model_in += sub_queries + '\n\n'
    model_in += doc_text
    model_in += f'###Examples:\n{ex.examples}\n\n'
    model_in += f'###Query:\n{query}'

    start_time = time.time()
    output = generate_output(model_in)
    generation_time = time.time() - start_time
    print(f"Generation Time: {generation_time} seconds")
    
    print(output)
    
    doc_and_examples = doc_text + '\n\nHere are some examples: \n' + ex.examples
    feedback = get_feedback(query, output, doc_and_examples)
    
    start_time = time.time()
    model_in2 = ex.feedback_prompt + '\n\n' + doc_text + f'###Examples:\n{ex.examples}\n\n' + f'###Output: {output}\n\n' + f'###Feedback: {feedback}\n\n' + f'###Query:\n{query}'
    output2 = generate_output(model_in2)
    feedback_generation_time = time.time() - start_time
    print(f"Final Output Generation Time: {feedback_generation_time} seconds")
    
    print(output2)
    
    print(f"Total time in pipeline: {retrieval_time + generation_time + feedback_generation_time}s")
    
    if 'unanswerable' in output2.lower() or 'unanswerable' in output.lower():
        output2 = 'Unanswerable'
        return {'Output': 'None'}

    try:
        return parser.function_to_json(output2)
    except Exception as e:
        print(f"Error: {e}")
        return {'Output': output2}

app = FastAPI()

# Allow requests from all origins during development
app.add_middleware(
    CORSMiddleware,
    # Update this with your frontend URL
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryInput(BaseModel):
    query: str


class QueryOutput(BaseModel):
    isResponse: bool
    text: str
    code: dict


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/api/query/", response_model=QueryOutput)
async def process_query_endpoint(query_input: QueryInput):
    # answerability = check_ans(query_input.query)
    # if answerability == 'Unanswerable':
    #     return {"isResponse": True, "text": "Sorry, I don't know the answer to that. Please try another question.", "code": {}}
    processed_text = process_query(query_input.query)
    pipeline_output = pipeline(query_input.query)
    if pipeline_output['Output'] == 'None':
        processed_text += '\n\nSorry! Looks like I cannot answer this request using the provided APIs.'
    return {"isResponse": True, "text": processed_text, "code": pipeline_output}
