from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import time
import requests

ASSISTANT_ID = 'asst_VSDOllgBf0GkI50zVnBJdJ5N'
RETRIEVER_ID = 'asst_xMGkLqbVNc1EYAg14AFxpEdr'
# TODO: Update this with your ngrok URL
NGROK_LINK = 'https://adb1-34-32-253-133.ngrok.io'

client = OpenAI()
thread = client.beta.threads.create()
ret_thread = client.beta.threads.create()

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
    # sys_prompt = 'You are a friendly chatbot. The user will either enquire or will have a natural conversation. Each query would be resolved by a different pipeline in the backend which will generate a sequence of API calls in json format. Your goal is to *just* generate brief text (upto 35 words) that precedes the actual answer. Do not make any assumptions but act as if you are looking into it.'
    
    # completion = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": sys_prompt},
    #         {"role": "user", "content": query}
    #     ]
    # )
    # return completion.choices[0].message.content

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
    )
    
    wait_on_run(run, thread)
    
    messages = client.beta.threads.messages.list(
        thread_id=thread.id,
        order='desc',
    )
    
    print(messages)
    idx, ite = -1, 0
    for i, msg in enumerate(messages):
        if msg.content[0].text.value == query:
            idx += i
    
    for msg in messages:
        if ite == idx:
            return msg.content[0].text.value
        ite += 1


def retriever(query: str):
    message = client.beta.threads.messages.create(
        thread_id=ret_thread.id,
        role="user",
        content=query
    )
    run = client.beta.threads.runs.create(
        thread_id=ret_thread.id,
        assistant_id=RETRIEVER_ID,
    )
    wait_on_run(run, ret_thread)
    messages = client.beta.threads.messages.list(
        thread_id=ret_thread.id,
        order='desc',
    )
    
    print(messages)
    idx, ite = -1, 0
    for i, msg in enumerate(messages):
        if msg.content[0].text.value == query:
            idx += i
    
    for msg in messages:
        if ite == idx:
            return {'tools': msg.content[0].text.value}
        ite += 1
        

def check_ans(query):
    payload = {"query": query}
    req = requests.post(
        NGROK_LINK+'/predict/', json=payload)
    return req.json()['text']
    
    
def pipeline(query: str):
    return retriever(query)

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
    answerability = check_ans(query_input.query)
    if answerability == 'Unanswerable':
        return {"isResponse": True, "text": "Sorry, I don't know the answer to that. Please try another question.", "code": {}}
    processed_text = process_query(query_input.query)
    pipeline_output = pipeline(query_input.query)
    return {"isResponse": True, "text": processed_text, "code": pipeline_output}
