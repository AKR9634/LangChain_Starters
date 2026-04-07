# TypedDict is only for type hinting and not for validation...
from typing import TypedDict

class Person(TypedDict):
    
    name: str
    age: int

new_person: Person = {'name':'akr', 'age':23}

print(new_person)