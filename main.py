from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


def process_query(query: str) -> str:
    # From GPT we will get the text for the query
    # For demonstration purposes, simply returning the processed query
    return f"Processed: {query}"


def pipeline(query: str):
    # Your ML pipeline logic goes here
    # For demonstration purposes, simply returning the a sample json
    return {
        "intent": {
            "name": 'account',
            "confidence": 0.9999999999999999,
        },
        "entities": {},
    }


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
    processed_text = process_query(query_input.query)
    pipeline_output = pipeline(query_input.query)
    return {"isResponse": True, "text": processed_text, "code": pipeline_output}
