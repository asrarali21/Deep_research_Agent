from fastapi import FastAPI

from api.user_query import router as user_query_router

app = FastAPI()


app.include_router(user_query_router)


@app.get("/")
def check():
    return "hello king working"





