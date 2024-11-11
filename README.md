# Campus Connect

Campus Connect is an intelligent chatbot system developed for Sacramento State University, designed to provide an efficient and responsive way for students, faculty, and staff to access campus information. Using advanced NLP, LLM, and RAG techniques, this chatbot enhances user engagement and access to university-related information.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

### Project Overview
Campus Connect was developed to streamline access to university information through a user-friendly chatbot, making it easier for students and staff to get answers to frequently asked questions.

- **Purpose**: To provide a chatbot-based solution for retrieving information related to Sacramento State University.
- **Context**: Developed as part of a project at Sacramento State University, leveraging LLMs to enhance interaction quality.
- **Scope**: Users can access information on campus tours, tuition, activity options, and job opportunities through intuitive conversation.

### Features
- **Message Handling**: Manages user and bot messages, handling both text responses and image-based replies.
- **Speech Recognition**: Allows users to input messages via voice with start/stop functionality.
- **Feedback Mechanism**: Includes star ratings and emoji feedback options for user responses.
- **Interactive Suggestions**: Context-aware suggestions guide the conversation.
- **Information Options**: Quick access to campus tours, tuition info, events, and job opportunities.
- **Hyperlink Parsing**: Detects URLs and emails in responses, turning them into clickable links.
- **Emoji Pop-Up**: Allows quick reactions with emojis after bot messages.
- **Editable User Messages**: Users can edit previously sent messages.
- **Image Processing**: Supports images in chat.
- **Loading Indicator**: Displays a loading animation while processing requests.
- **Feedback Submission**: User feedback is collected and sent to the backend.

### Demo
<img width="361" alt="image" src="https://github.com/user-attachments/assets/dcfb84fd-f0c8-4b06-900c-f26054972309">
<img width="577" alt="image" src="https://github.com/user-attachments/assets/75ca1fc6-4269-4fa5-9544-e5a401f2166c">
<img width="453" alt="image" src="https://github.com/user-attachments/assets/91c0d93d-248f-422c-b640-df13748efce4">
<img width="361" alt="image" src="https://github.com/user-attachments/assets/e6baf7b2-e63c-49dc-986a-c1ea361df909">

### Dataset
- **Data Source**: University-specific data, including general information, FAQs, and student resources, was collected and curated to form the chatbot's knowledge base.
- **Data Format**: A combination of structured data (CSV files for FAQs and contacts) and unstructured text documents (university-related information).
- **Preprocessing**: Data was preprocessed to improve the chatbot’s response quality. Techniques included:
  - **Tokenization**: Breaking down text into manageable tokens.
  - **Text Cleaning**: Removing irrelevant characters, stopwords, and special symbols.
  - **Encoding**: Using sentence embeddings to enable efficient retrieval and ranking of responses.
    <img width="799" alt="image" src="https://github.com/user-attachments/assets/37ab1b5a-5e2f-4976-bcd8-db518da6a766">


### Methodology
1. **Data Collection**:
   - Created a dataset specific to Sacramento State University, focusing on FAQs and commonly requested resources.
   - Used both structured (CSV) and unstructured data (text documents).

2. **Model Architecture**:
   - Leveraged LLMs, Retrieval-Augmented Generation (RAG), and NLP techniques for accurate, context-aware response generation.
   - Integrated FAISS-based vector search for hybrid search capabilities and applied reciprocal rank fusion for result ranking.
   -  - **CNN for Images and Maps**: Used Convolutional Neural Networks (CNN) to process images of campus maps and building structures, enabling location-based queries and visually assisted responses.
   - **Speech-to-Text Integration**: Enabled voice input using speech-to-text processing to allow users to interact with the chatbot via spoken queries, which are converted to text for processing.
<img width="176" alt="image" src="https://github.com/user-attachments/assets/d6e8fe64-a83b-44a5-820a-4704a14a6104">


3. **Training and Optimization**:
   - Fine-tuned the chatbot to handle queries in context, applying techniques like “lost in the middle” ordering and query expansion.
   - Used `CharacterTextSplitter` to manage chunk sizes effectively and semantic ranking to ensure response relevance.

4. **Evaluation**:
   - Tested on various cases, including follow-up and contextual questions, to assess accuracy and response quality.

### Results
- **User Satisfaction**: Positive feedback on features like interactive suggestions and voice input.
- **Accuracy**: High accuracy in answering FAQs and general campus queries.
- **Performance**: Efficient query handling and response generation through hybrid search.
- **Error Analysis**: Minor issues in multi-step queries were addressed through model fine-tuning.
- <img width="555" alt="image" src="https://github.com/user-attachments/assets/980a328b-b8a0-4bf6-9d83-c9c07816eaf0">


### Technologies Used
- **Backend**: Python, Flask, TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript, React
- **Machine Learning**: OpenAI, FAISS, NLP
- **Libraries and Tools**:  LLM, RAG, CharacterTextSplitter

### References
- [DeepSet Documentation on Generative Question Answering](https://docs.cloud.deepset.ai/docs/generative-question-answering)
- [Google Developers: Convolutional Neural Networks](https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks)
- [Haystack by Deepset: NLP Framework for Question Answering](https://haystack.deepset.ai/)
- [OpenAI Documentation: Large Language Models (LLMs)](https://platform.openai.com/docs/)
- Zhu, J., Chen, X., Li, P., & Liu, W. (2020). Design of a face recognition system based on convolutional neural network (CNN). *Engineering, Technology & Applied Science Research, 10*(3), 5608-5612.
- [BM25 Algorithm for Information Retrieval](https://en.wikipedia.org/wiki/Okapi_BM25)
- [FAISS: Facebook AI Similarity Search](https://faiss.ai/)






