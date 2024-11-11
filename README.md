Campus Connect
Campus Connect is an intelligent chatbot system developed for Sacramento State University, designed to provide an efficient and responsive way for students, faculty, and staff to interact with campus information. By leveraging advanced NLP, LLM, and RAG techniques, this chatbot system enhances user engagement and access to university-related information.

Table of Contents
Project Overview
Features
Demo
Installation
Usage
Methodology
Results
Technologies Used
Contributing
License
Acknowledgments
Project Overview
Purpose: Campus Connect was created to streamline access to university information through a user-friendly chatbot, making it easier for students and staff to get answers to frequently asked questions.
Context: The project was developed at Sacramento State University, incorporating LLMs to improve chatbot interaction and comprehension.
Scope: The chatbot provides access to university-related resources, such as campus tours, tuition information, activity options, and on-campus job opportunities.
Features
Message Handling: Supports both user and bot messages, managing text responses and image-based replies.
Speech Recognition: Allows users to input messages via voice with start/stop functionality.
Feedback Mechanism: Users can provide feedback through a star rating and emoji feedback for bot responses.
Interactive Suggestions: Context-aware suggestions are offered to guide the conversation.
Information Options: Pre-configured buttons give access to information like campus tours, tuition, upcoming events, and job opportunities.
Hyperlink Parsing: Automatically detects URLs and email addresses, converting them into clickable links.
Emoji Pop-Up: Allows quick user reactions with emojis after each bot message.
Editable User Messages: Users can edit their previously sent messages.
Image Processing: Supports image display as part of the conversation.
Loading Indicator: Shows a loading animation while the bot processes requests.
Feedback Submission: User feedback is sent to the backend for analysis.
Demo
Include screenshots, gifs, or a link to a live demo if available.

Installation
To set up the Campus Connect project locally, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/campus-connect.git
Navigate into the project directory:

bash
Copy code
cd campus-connect
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up environment variables:

Create a .env file and add required environment variables as per .env.example.
Run the project:

bash
Copy code
python main.py  # Replace with the actual main file if different
Usage
Starting the server:

bash
Copy code
python app.py
Accessing the chatbot:

Open a browser and navigate to http://localhost:5000 to interact with the chatbot.
Features in Use:

Explore features like image-based replies, voice input, and context-aware suggestions.
Methodology
Data Collection:

University-specific data was collected, including general information and FAQs, to populate the chatbot’s knowledge base.
A combination of text-based documents and structured data formats was used.
Model Architecture:

The chatbot uses a combination of LLMs, RAG (Retrieval-Augmented Generation), and NLP techniques to process and respond to queries.
BM25 and FAISS-based vector search methods were incorporated to retrieve relevant results, applying reciprocal rank fusion for result ranking.
Training and Optimization:

Fine-tuning was performed to improve the chatbot’s responses, including context handling, “lost in the middle” ordering, and query expansion.
Text splitting techniques with CharacterTextSplitter and semantic ranking were applied to improve response relevance and manage chunk sizes effectively.
Evaluation:

Various cases, such as follow-up questions and contextual queries, were tested to ensure the chatbot’s accuracy and robustness.
Results
User Satisfaction: Positive feedback was gathered based on the user experience features, such as interactive suggestions and voice input.
Accuracy: The chatbot achieved high accuracy in answering FAQ-type questions about campus resources and events.
Performance: Efficient query handling and response generation, with improvements in relevance achieved through hybrid search methods.
Error Analysis: Minor misinterpretations in complex, multi-step queries were addressed through model fine-tuning.
Technologies Used
Backend: Python, Flask, TensorFlow/Keras
Frontend: HTML, CSS, JavaScript, React
Database: Firebase, MySQL
Machine Learning: OpenAI, BM25, FAISS, NLP
Libraries and Tools: OpenCV, LLM, RAG, CharacterTextSplitter






