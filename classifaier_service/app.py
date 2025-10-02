from fastapi import FastAPI
from pydantic import BaseModel
from classifaier_service.schemas import OurUser

api = FastAPI()

class AdditionInput(BaseModel):
    numbers: tuple[float, ...]

all_users = {}

@api.get("/")
def say_hello(name: str | None = None):
    if not name:
        name = "Guest"
    return f"Hello {name}"

# @api.get("/add")
# def add(a,b):
#     return int(a)+int(b)

@api.post("/add")
def add_numbers(data: AdditionInput):
    return sum(data.numbers)

@api.post("/add_user")
def add_user(user: OurUser):
    user_data = user.model_dump()
    user_data["id"] = len(all_users)
    all_users[user_data["id"]] = user_data
    return user_data
    

@api.get("/user_age_avg")
def users_age_average():
    if not all_users:
        return {"average_age": 0}

    total_age = 0
    for user_data in all_users.values():
        total_age += user_data.get("age", 0)  

    average_age = total_age / len(all_users)
    return {"average_age": average_age}
    

@api.post("/user_age")
def double_the_age(user: OurUser):
    return user.age*2