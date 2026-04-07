from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text_splitter = SemanticChunker(
    embedding, breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample = """
Uttarakhand is a state located in the northern part of India. It was carved out of the state of Uttar Pradesh in the year 2000. The state is known for its scenic beauty, including the Himalayan mountain ranges, rivers, and lush forests. The capital of Uttarakhand is Dehradun, and other major cities include Haridwar, Rishikesh, and Nainital.
Virat Kohli is an Indian cricketer and one of the most successful batsmen in the world. He was born on November 5, 1988, in Delhi, India. Kohli made his debut for the Indian national cricket team in 2008 in One Day Internationals (ODIs) and later played in Test matches and T20 internationals.
He is known for his aggressive batting style, consistency, and fitness. Kohli has scored thousands of runs across all formats of international cricket and has received numerous awards, including the prestigious Sir Garfield Sobers Trophy for ICC Cricketer of the Year.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)