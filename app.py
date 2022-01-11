from fastapi import FastAPI
import pickle
import asyncio
async def get_model(path:str="moidel.pkl"):
    with open(path,"rb") as f:
        model=pickle.load(f)
    return model
app=FastAPI()
@app.get("/")
async def home():
    return {"message":"Welcome to the hellow world"}

@app.get("/predict")
async def predict(experience:float):
    model=await get_model()
    prediction=model.predict([[experience]])
    return {"salary": prediction[0][0]}

