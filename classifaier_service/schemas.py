from pydantic import BaseModel, Field

class OurUser(BaseModel):
    age : int = Field(ge=16, le=120)
    
