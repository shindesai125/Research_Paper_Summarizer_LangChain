from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    temperature=0.3,
    # max_new_tokens=200
)



model = ChatHuggingFace(llm=llm)

st.header("Research Paper Summarizer using openai/gpt-oss-20b")

paper_input = st.selectbox("Select Research Paper", ['Sex Education', 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', 'GPT-3: Language Models are Few-Shot Learners', 'Diffusion Models Beat GANs on Image Synthesis'])

style_input = st.selectbox("Select Summary Style", ['Begginer Friendly', 'Technical', 'Code-Oriented', 'Mathematical'])

length_input = st.selectbox(
    "Select the Length of Summary",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)


# template
template = PromptTemplate(template = """Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}
Explanation Length: {length_input}

Mathematical Details:

Include relevant mathematical equations if present in the paper.

Explain the mathematical concepts using simple, intuitive code snippets where applicable.

Analogies:

Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.""",
input_variables = ['paper_input','style_input','length_input']
)


# fill placeholders
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input})


if st.button("Generate Summary"):
    result = model.invoke(prompt)
    st.write("### Summary:",
             result.content)
