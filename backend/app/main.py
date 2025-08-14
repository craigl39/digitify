# backend/main.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Digitify API is up and running!"}
