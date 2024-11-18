from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from schemas import Title_Comment, Comment
from input_handler import handle_input_with_title, handle_input_no_title

app = FastAPI()

# # Set up CORS
# allowed_origins = ['http://localhost:4567']
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[allowed_origins],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

@app.post("/api/v1/predict/title")
def read_root(req: Title_Comment):
    res = handle_input_with_title(req)[0]
    if res == 0:
        res = 'Negative'
    elif res == 1:
        res = 'Neutral'
    else:
        res = 'Positive'

    return {'Sentiment': res}

@app.post("/api/v1/predict/notitle")
def read_root(req: Comment):
    res = handle_input_no_title(req)[0]
    if res == 0:
        res = 'Negative'
    elif res == 1:
        res = 'Neutral'
    else:
        res = 'Positive'

    return {'Sentiment': res}
