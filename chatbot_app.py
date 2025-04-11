
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import faiss
import pandas as pd

# Load pre-trained models and data
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
fallback_model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_flan_t5_base")
fallback_tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_flan_t5_base")

# Load FAISS index
index = faiss.read_index("/kaggle/working/faiss_index.index")

# Load dataset questions and answers
data = pd.read_csv('/kaggle/input/medicaldata/finaldata.csv')
questions = data['question'].tolist()
answers = data['answer'].tolist()

# Fallback function
def generate_fallback_answer(query):
    inputs = fallback_tokenizer(
        query, return_tensors="pt", max_length=128, truncation=True, padding="max_length"
    )
    outputs = fallback_model.generate(
        inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True
    )
    return fallback_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Query pipeline
threshold = 0.8
def query_pipeline(user_query):
    query_embedding = retrieval_model.encode([user_query], convert_to_numpy=True)
    k = 1
    distances, indices = index.search(query_embedding, k)
    distance = distances[0][0]
    retrieved_answer = answers[indices[0][0]]

    if distance > threshold:
        return generate_fallback_answer(user_query)
    else:
        return retrieved_answer

# Streamlit app
def main():
    st.title("Doctor GPT 🩺")
    st.write("Ask any medical query and let the chatbot assist you!")

    # User input
    user_query = st.text_input("Enter your medical question:")

    if st.button("Get Answer"):
        if user_query:
            response = query_pipeline(user_query)
            st.success(f"**Response:** {response}")
        else:
            st.error("Please enter a question!")

if __name__ == "__main__":
    main()
