**Problem Statement**: 

CampusConnect an intelligent chat program for Sacramento State University. The objective is to optimise the campus experience for users of Sac State University by providing them with assistance with departmental information, job postings, event calendars, and all other relevant information about Sac State University.

**Requirements** :

+Google Colab: Utilized as the primary development environment for coding and experimentation.

+Web Scraping with BeautifulSoup (bs4): Utilized for extracting information from web pages for real-time data updates.

+Large Language Models (LLMs): Leveraged from OpenAI for advanced Natural Language Processing (NLP) capabilities.

+OpenAI API Key Required for accessing OpenAI's LLMs and other services through their API.

+Language Model (e.g., TensorFlow Hub): Possibly used for fine-tuning and deploying the LLM for improved understanding and response generation.

+TensorFlow Hub: Potentially used for accessing pre-trained language models or embeddings for NLP tasks.

**WorkFlow**:
The workflow consists of the following steps:
****Data Extraction with BeautifulSoup (bs4)****:
Initially, hyperlinks are extracted from URLs using BeautifulSoup (bs4).
This step aims to gather relevant data sources for subsequent processing.
****Text Embedding with TensorFlow Hub****:
The extracted hyperlinks are utilized to retrieve text data.
TensorFlow Hub is employed to embed the text data into high-dimensional vector representations.
This process enhances the semantic understanding of the text, facilitating more effective information retrieval.
****Document Similarity Search****:
The similarity_search(query) function is employed to extract similar data based on the user's query.
Utilizing similarity scores as metrics, relevant documents are identified for further processing.
This step ensures that the retrieved documents are contextually aligned with the user's query.
****QA Pipeline with OpenAI API and LLM****:
Utilizing the OpenAI API key, a Question Answering (QA) pipeline is instantiated.
The text data is segmented into chunks, and a document search is conducted using LangChain.
This step ensures that relevant documents containing potential answers are identified efficiently.
****Query Processing and Response Generation****:
User queries, such as "How to contact registration office?", are retrieved for processing.
Based on the retrieved documents and the query, the QA pipeline generates contextually relevant responses.
The responses are presented to the user via the CampusConnect chat interface.

**Getting Started**:
Follow the instructions below to set up and run CampusConnect on your local machine:
****Clone the repository****: git clone https://github.com/yourusername/campus-connect.git.
****Install the required dependencies****: pip install -r requirements.txt.
****Obtain an API key from OpenAI and set it as an environment variable****: export OPENAI_API_KEY="your-api-key"
****Run the main application file****: python main.py.






