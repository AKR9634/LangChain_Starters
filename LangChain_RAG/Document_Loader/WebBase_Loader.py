from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

url = "https://www.amazon.in/OnePlus-Snapdragon%C2%AE-7400mAh-Personalised-Game-Changing/dp/B0FZSWZZW2/ref=sr_1_2?crid=32ROY5PNEN726&dib=eyJ2IjoiMSJ9.a3BtxWjZVK7jGpeWmFQpQMDkEopSlSAHplr-gkl1iSUjv1fJneF85gDDQGg7vQKUL9kyujCD_fDd7Bl9echMCVNB0UeCWf_wPoBuWSVyfft7uDOfjoP3flUtdD0aELxhcZl_FhqZhLRfnBKCigRRkOEbURsruviskky9AIzr8L7FHxH7Z76ihf055Uv7egi8lP_9_5Y5bWCxUoeSwovGyn48br_t9y63sJ1jWxWrJu4.R9YIclwNwFG-eWqAtiu8t8p1l33uaWUrM6tpoSrQPLY&dib_tag=se&keywords=one%2Bplus%2B13s&qid=1775305523&sprefix=one%2Bplus%2B13s%2Caps%2C426&sr=8-2&th=1"
loader = WebBaseLoader(url)

docs = loader.load()



