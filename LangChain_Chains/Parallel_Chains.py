from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task = "text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)


model1= ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Summarize the content of the following text in about 30 words!!! {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Provide 5 one line quiz questions on the following text: {text}",
    input_variables=['text']
)

parser = StrOutputParser()

prompt3 = PromptTemplate(
    template="Merge the following summary: {summary} and the quiz: {quiz} \n into a single document together...",
    input_variables=['summary', 'quiz']
)

chain1 = prompt1 | model1 | parser
chain2 = prompt2 | model2 | parser

parallel_chain = RunnableParallel({
    'summary' : chain1,
    'quiz' : chain2
})

merged_chain = prompt3 | model1 | parser

chain = parallel_chain | merged_chain

text = """
im Corbett National Park is a national park in the Nainital district of the state of Uttarakhand, India. It was established in 1936 as the country's first national park and is named in honour of Jim Corbett, who had played a leading role in its establishment. It was the first to come under the Project Tiger initiative and encompasses an area of 520.8 km2 (201.1 sq mi) consisting of hills, riverine belts, marshy depressions, grasslands and a large lake at an elevation range of 400–1,220 m (1,300–4,000 ft). It receives rainfall from July to September. Almost 73% of the national park is covered by dense moist deciduous forest dominated by Shorea robusta, peepal, rohini and mango trees among 110 tree species and 617 different plant species. Grasslands cover about 10% of its total area. It harbours 50 mammal species, 580 bird species and 25 reptile species. The increase in tourist activities continues to present a serious challenge to the park's ecological balance.

History
Some areas of the park were formerly part of the princely state of Tehri Garhwal.[2] The forests were cleared by the Uttarakhand Forest Department to make the area less vulnerable to Rohilla invaders.[2] The Raja of Tehri formally ceded a part of his princely state to the East India Company in return for their assistance in ousting the Gurkhas from his domain.[2] The Buksas—a tribe from the Terai—settled on the land and began growing crops, but in the early 1860s they were evicted with the advent of British rule.[2]

Efforts to save the forests of the region began in the 19th century under Major Ramsay, the British officer who was in-charge of the area during those times. The first step in the protection of the area began in 1868 when the British forest department established control over the land and prohibited cultivation and the operation of cattle stations.[3] In 1879 these forests were constituted into a reserve forest where restricted felling was permitted.

In the early 1900s, several Britishers, including E. R. Stevans and E. A. Smythies, suggested the setting up of a national park on this soil. The British administration considered the possibility of creating a game reserve there in 1907.[3] It was only in the 1930s that the process of demarcation for such an area got underway. A reserve area known as Hailey National Park covering 323.75 km2 (125.00 sq mi) was created in 1936, when Sir Malcolm Hailey was the Governor of United Provinces, and Asia's first national park came into existence.[4] Hunting was not allowed in the reserve, only timber cutting for domestic purposes was permitted. Soon after the establishment of the reserve, rules prohibiting the killing and capture of mammals, reptiles and birds within its boundaries were passed.[4]

The reserve was renamed Ramganga National Park in 1954–1955 and was again renamed in 1955–1956 to Corbett National Park after author and naturalist Jim Corbett.[4] The park fared well during the 1930s under an elected administration. But during the Second World War, it suffered from excessive poaching and timber cutting. Over time, the area in the reserve was increased to 797.72 km2 (308.00 sq mi) were added in 1991 as a buffer zone to the Corbett Tiger Reserve.[4] The 1991 addition included the entire Kalagarh forest division, assimilating the 301.18 km2 (116.29 sq mi) area of Sonanadi Wildlife Sanctuary as a part of the Kalagarh division.[4] It was chosen in 1974 as the location for launching the Project Tiger wildlife conservation project.[5] The reserve is administered from its headquarters in the Nainital district.[3]

Corbett National Park is one of the thirteen protected areas covered by the World Wide Fund For Nature under their Terai Arc Landscape Program.[6] The program aims to protect three of the five terrestrial flagship species, the tiger, the Asian elephant and the Indian rhinoceros by restoring wildlife corridors to link 13 protected areas of Nepal and India and to enable wildlife migration.[6
"""

result = chain.invoke({'text' : text})

print(result)

chain.get_graph().print_ascii()