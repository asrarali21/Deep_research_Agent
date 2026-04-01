from fastapi import APIRouter 
from pydantic import BaseModel

router = APIRouter()


class Queryrequest(BaseModel):
    query:str


@router.post("/start")
def user_query(request : Queryrequest):

    return request.query
    