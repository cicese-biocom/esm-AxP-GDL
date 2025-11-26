from pydantic.v1 import BaseModel


class BaseDataTransferObject(BaseModel):
    class Config:
        arbitrary_types_allowed = True