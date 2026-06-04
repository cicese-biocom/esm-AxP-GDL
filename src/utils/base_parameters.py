from pydantic.v1 import BaseModel


class BaseParameters(BaseModel):
    class Config:
        arbitrary_types_allowed = True