import streamlit as st
import uuid
from utils import *

# Create session state for the first tab
if 'unique_id_tab1' not in st.session_state:
    st.session_state["unique_id_tab1"] = ''

# Create session state for the second tab
if 'unique_id_tab2' not in st.session_state:
    st.session_state["unique_id_tab2"] = ''

def main():
    st.set_page_config(page_title="RESUME SCREENING APP")
    st.title("AI Resume Screening System 🤖 📑")
    st.subheader("Helping you with resume screening 📝")

    uploaded_files = None

    def docembed(uploaded_files, tab_key):
        # Use different session state variables for each tab
        docs = create_doc(uploaded_files, st.session_state[f"unique_id_{tab_key}"])
        embeddings = create_embedding_load_data()

        return docs, embeddings

    tabs = st.tabs(["Analyse Resumes", "Upload Resumes"])

    with tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            uploaded_files = st.file_uploader("Upload Resume here 📎 (Only PDF files allowed) :-", type=["pdf"], accept_multiple_files=True)
            submit_btn = st.button("Start Pushing to DB")
            if submit_btn:
                with st.spinner("Wait It's in progress........"):
                    # Use the session state variable for the second tab
                    st.session_state["unique_id_tab2"] = uuid.uuid4().hex
                    
                    docs, embeddings = docembed(uploaded_files=uploaded_files, tab_key="tab2")
                    push_to_pinecone("56ea0e97-0ca4-4db7-b74c-4e60dd726bf5", "gcp-starter", "resumeanalyser", embeddings, docs)
                st.success("Successfully Uploaded to Pinecone.")

        with col2:
            st.warning("Delete data from Pinecone 🚨")    
            delete_button = st.button("Delete")
            if delete_button:
                with st.spinner("Deleting Data."):
                    delete_from_pinecone("56ea0e97-0ca4-4db7-b74c-4e60dd726bf5", "gcp-starter", "resumeanalyser") 
                st.success("Successfully Deleted from Pinecone.")

    with tabs[0]:
        job_description = st.text_area("Please provide Job Description")
        document_count = st.text_input("Number of resumes to be retrieved :-", key="2")
        
        # Use the session state variable for the first tab
        docs, embeddings = docembed(uploaded_files=uploaded_files, tab_key="tab1")

        submit_btn_results = st.button("Start analyzing")  
        
        if submit_btn_results:
            with st.spinner("Wait It's in progress........"):
                st.session_state["unique_id_tab1"] = uuid.uuid4().hex
                relevant_docs = similar_doc(job_description, document_count, "56ea0e97-0ca4-4db7-b74c-4e60dd726bf5",
                                            "gcp-starter", "resumeanalyser", embeddings, st.session_state["unique_id_tab1"], docs)
                # st.write(relevant_docs)

                for doc in relevant_docs:
                    content = doc[0]
                    score = doc[1]
                    score = score * 100
                    with st.expander("Show Score:-", expanded=True):
                        st.info("**match score** :" + str(score)+"%")
                        summary = get_summary(doc[0])
                        st.write(summary)

            st.success("Hope it was helpful.")

if __name__ == "__main__":
    main()
