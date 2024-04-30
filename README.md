## Problem Statement :  

CampusConnect an intelligent chat program for Sacramento State University. The objective is to optimise the campus experience for users of Sac State University by providing them with assistance with departmental information, job postings, event calendars, and all other relevant information about Sac State University.

## Requirements

- *Google Colab*: Utilized as the primary development environment for coding and experimentation.    

- *Web Scraping with BeautifulSoup (bs4)*: Utilized for extracting information from web pages for real-time data updates.
                                                                                                                                                                                         
- *Large Language Models (LLMs)*: Leveraged from OpenAI for advanced Natural Language Processing (NLP) capabilities.
                                                                                                                                                                                         
- *OpenAI API Key*: Required for accessing OpenAI's LLMs and other services through their API.

- *Language Model (e.g., TensorFlow Hub)*: Possibly used for fine-tuning and deploying the LLM for improved understanding and response generation.
                                                                                                                                                                                         
- *TensorFlow Hub*: Potentially used for accessing pre-trained language models or embeddings for NLP tasks.

## Source of Truth :

The data used in this project is sourced from Sacramento State University (CSUS) URLs, which serve as the authoritative source of information for various departments, courses, and resources. The following URLs are examples of the source of truth for specific data categories:

                                https://catalog.csus.edu/courses-a-z/csc/
                                https://www.csus.edu/college/engineering-computer-science/computer-science/meet-us/

By utilizing these URLs as the source of truth, we ensure that the data used in the project remains accurate, up-to-date, and aligned with official university information.





## WorkFlow :

The workflow consists of the following steps:

****Data Extraction with BeautifulSoup (bs4)****:

         -Initially, hyperlinks are extracted from URLs using BeautifulSoup (bs4).
         -This step aims to gather relevant data sources for subsequent processing.

****Text Embedding with TensorFlow Hub****:

         -The extracted hyperlinks are utilized to retrieve text data.
         -TensorFlow Hub is employed to embed the text data into high-dimensional vector representations.
         -This process enhances the semantic understanding of the text, facilitating more effective information retrieval.

****Document Similarity Search****:

        -The similarity_search(query) function is employed to extract similar data based on the user's query.
        -Utilizing similarity scores as metrics, relevant documents are identified for further processing.
        -This step ensures that the retrieved documents are contextually aligned with the user's query.

****QA Pipeline with OpenAI API and LLM****:

      -Utilizing the OpenAI API key, a Question Answering (QA) pipeline is instantiated.
     -The text data is segmented into chunks, and a document search is conducted using LangChain.
     -This step ensures that relevant documents containing potential answers are identified efficiently.

****Query Processing and Response Generation****:

    -User queries, such as "How to contact registration office?", are retrieved for processing.
    -Based on the retrieved documents and the query, the QA pipeline generates contextually relevant responses.
    -The responses are presented to the user via the CampusConnect chat interface.

## Getting Started :

Follow the instructions below to set up and run CampusConnect on your local machine:

****Clone the repository****: git clone https://github.com/yourusername/campus-connect.git.

****Install the required dependencies****: pip install -r requirements.txt.

****Obtain an API key from OpenAI and set it as an environment variable****: export OPENAI_API_KEY="your-api-key"

****Run the main application file****: python main.py.


## Tech Stack :

- **OpenAI**: v1.24.0
- **LangChain**: v0.1.16
- **Python**: v3.10.12


## Future Work :

- *Adding More Data*: Expand the dataset by incorporating data from other relevant URLs related to Sacramento State University. This could include information from departmental pages, faculty profiles, or campus news sources.

- *Fine-Tuning Data for Accuracy*: Implement techniques such as  model fine-tuning to improve the accuracy of the project's outputs. This involves refining the existing data sources and optimizing the models used in the project to ensure more precise and reliable results.

- *Building a Web Application*: Develop a web application using HTML, CSS, and JavaScript to provide a user-friendly interface for accessing and interacting with the project's data. The application could include features such as search functionality, filtering options, and visualization tools to enhance the user experience.

## References

- [Building Domain-Specific LLMs: How to Work with Your Own Data](https://dreamproit.com/blog/2024-02-06-building-domain-specific-LLMs-how-to-work-with-your-own-data/index.html)

- [TAIPY Documentation: Chatbot Tutorials](https://docs.taipy.io/en/release-3.0/knowledge_base/tutorials/chatbot/)

- [The Future of Language Models: The Rise of Domain-Specific Expertise](https://www.linkedin.com/pulse/future-language-models-rise-domain-specific-expertise-magnuszewski-stwoe/)

## Additional Resources and Learnings

- [Haystack Tutorials](https://haystack.deepset.ai/tutorials): Explore tutorials on using the Haystack framework for building question-answering systems, semantic search, and more.

- [Comprehensive Guide for Building RAG-based LLM Applications (Part 1)](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1): Learn about building RAG (Retriever-Reader-Generator)-based Large Language Model (LLM) applications, including architecture, components, and best practices.

- [LangChain Crash Course](https://codebasics.io/resources/langchain-crash-course): Dive into a crash course on LangChain, covering fundamental concepts, usage, and examples for implementing language models and NLP tasks.

          






