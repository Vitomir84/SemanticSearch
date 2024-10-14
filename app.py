import streamlit as st
from models.prompt_search_engine import PromptSearchEngine
from models.data_reader import load_prompts_from_jsonl

# Cache the prompts data to avoid reloading every time
@st.cache_data
def load_prompts():
    prompt_path = "data/prompts_data.jsonl"
    return load_prompts_from_jsonl(prompt_path)

# Cache the search engine initialization
@st.cache_resource
def get_search_engine():
    search_engine = PromptSearchEngine()
    prompts = load_prompts()
    search_engine.add_prompts_to_vector_database(prompts)
    return search_engine

# Initialize search engine only once
search_engine = get_search_engine()

# Streamlit App Interface
st.title("Prompt Search Engine")
st.write("Search for similar prompts using the local search engine.")

# Input for the user's prompt
query_input = st.text_input("Enter your prompt:")

# Number of similar prompts to retrieve (k)
k = st.number_input("Number of similar prompts to retrieve:", min_value=1, max_value=10, value=3)

# Button to trigger search
if st.button("Search Prompts"):
    if query_input:
        print(f'Search engine is searching the most similar prompts for query {query_input}')
        similar_prompts, distances = search_engine.most_similar(query_input, top_k=k)
        print(f'Those are: {similar_prompts}, {distances}')

        # Format and display search results
        st.write(f"Search Results: ")
        for i, (prompt, distance) in enumerate(zip(similar_prompts, distances)):
            st.write(f"{i+1}. Prompt: {prompt}, Distance: {distance}")
            print(f'Those are: {prompt}, {distance}')
    else:
        st.error("Please enter a prompt.")

# Additional functionality for vector similarity
st.write("---")
st.write("### Vector Similarities")

if st.button("Retrieve All Vector Similarities"):
    if query_input:
        query_embedding = search_engine.model.encode([query_input])  # Encode the prompt to a vector
        all_similarities = search_engine.cosine_similarity(query_embedding, search_engine.index)
        st.write(f"Vector Similarities: {all_similarities}")
    else:
        st.error("Please enter a prompt.")