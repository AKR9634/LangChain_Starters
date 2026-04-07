from pydantic import BaseModel, field_validator, model_validator, Field, EmailStr
from typing import Optional

class Student(BaseModel):

    name: str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description="A decimal value representing the CGPA of the student!!!")

new_student = {'name':'akr', 'age':23, 'email':'akr@gmail.com'}
student = Student(**new_student)

print(student)