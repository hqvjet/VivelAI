from pydantic import BaseModel

class Title_Comment(BaseModel):
    title: str
    comment: str

class Comment(BaseModel):
    comment: str
