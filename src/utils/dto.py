from pydantic.v1 import BaseModel


class DTO(BaseModel):
    class Config:
        arbitrary_types_allowed = True