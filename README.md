# Chatbot Backend

This project is made using FastAPI. This serves as a backend to the frontend chatbot.

**Note:** We consider updation or deletion as a new few, so the extra overhead should be counted as one-time setup.

## Run Locally

Go to the project directory

```bash
  cd chatbot-backend
```

Install dependencies

```bash
  pip install -r requirements.txt
```

```bash
   conda create --name myenv
   conda activate myenv
   conda env update --name myenv --file environment.yml --prune
```

Add your OpenApi key in main.py

```python
  OPENAPIKEY = 'XXXXXXXXXXXXXXXXXX'
```

Start the server

```bash
  uvicorn main:app --reload
```


## API Reference

#### Process the user query and sends the output

```http
  POST /api/query
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `query` | `string` | **Required**. The user query from the frontend |




## Tech Stack

**Server:** FastAPI, OpenAPI, Langchain

**Database:** ChromaDB

