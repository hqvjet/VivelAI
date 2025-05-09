from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import Title_Comment, Comment
from input_handler import handle_input_using_E2VPhoBERT, handle_input_using_E2TPhoBERT

app = FastAPI()

# # Set up CORS
allowed_origins = ['http://localhost:3000']
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/E2V-PhoBERT/predict")
def read_root(req: Comment):
    res = handle_input_using_E2VPhoBERT(req)[0]
    if res == 0:
        res = 'Negative'
    else:
        res = 'Positive'

    return {'Sentiment': res}

@app.post("/api/v1/VED-PhoBERT/predict")
def read_root(req: Comment):
    res = handle_input_using_E2TPhoBERT(req)[0]
    if res == 0:
        res = 'Negative'
    else:
        res = 'Positive'

    return {'Sentiment': res}
