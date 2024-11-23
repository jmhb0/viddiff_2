"""
python -m ipdb scripts/20241027_structured_outputs.py
"""
import ipdb
if 0:
	from pydantic import BaseModel, Field
	from typing import List, Optional

	class DataModel(BaseModel):
	    # For first list where second item is optional
	    lst: List[Optional[str]] = Field(..., min_items=2, max_items=2)
	    
	    # For second list where second item is optional
	    lst_optional: List[Optional[int]] = Field(default=None, min_items=3)    

	ipdb.set_trace()
	pass
	x = DataModel(lst=["a","b"])


if 0: 
	from pydantic import BaseModel, Field
	from openai import OpenAI
	import ipdb

	client = OpenAI()
	class Response(BaseModel):
	    answer: list[str] = Field( min_items=2, max_items=2)

	completion = client.beta.chat.completions.parse(
	    model="gpt-4o-2024-08-06",
	    messages=[
	        {"role": "system", "content": "You are a helpful assistant."},
	        {"role": "user", "content": "What is the biggest animal in one words"}
	    ],
	    response_format=Response,
	)

	response = completion.choices[0].message.parsed
	ipdb.set_trace()
	pass

from openai import OpenAI
from pydantic import BaseModel

class ListItem(BaseModel):
    item: str

class ListResponse(BaseModel):
    myList: list[ListItem]

client = OpenAI()

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Provide a list of items."},
        {"role": "user", "content": "Give me a list of 3 to 5 items."}
    ],
    response_format=ListResponse,
    response_format_options={
        "type": "json_schema",
        "json_schema": {
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "myList": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 5
                    }
                },
                "required": ["myList"],
                "additionalProperties": False
            }
        }
    }
)

list_response = completion.choices[0].message.parsed
print(list_response.myList)