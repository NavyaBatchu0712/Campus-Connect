api_key = ' 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'

# Use the API key in your OpenAI LLM setup
from langchain.llms import OpenAI

# Initialize the OpenAI LLM with the API key
llm = OpenAI(api_key=api_key)


import os

# Set the OpenAI API key as an environment variable in Python
os.environ['OPENAI_API_KEY'] = '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'
# Access the API key from the environment variable
api_key = os.getenv('OPENAI_API_KEY')

# Verify the key is set
print(api_key)  # Should print the API key

# Use the API key in your OpenAI LLM setup
from langchain.llms import OpenAI

# Initialize the OpenAI LLM with the API key
llm = OpenAI(api_key=api_key)



import os
from langchain.llms import OpenAI

# Load the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if the key is loaded correctly
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

# Use the API key to initialize the OpenAI LLM
llm = OpenAI(api_key=api_key)


import zipfile
import os

# Define the path to your zip file
zip_file_path = '/content/Sac State Data.zip'




import mimetypes

# Check the file type
mime_type, _ = mimetypes.guess_type(zip_file_path)
print(f"File type: {mime_type}")






import os

# Define the folder path
folder_path = '/Users/navyakrishnabatchu/Desktop/Chatbot/chatbot-ui/Campus-Connect/chatbot-frontend/Sac State Data'

# List the files in the folder
pdf_files = os.listdir(folder_path)
print("PDF Files in the folder:", pdf_files)


from langchain.document_loaders import PyMuPDFLoader

# Initialize a list to hold the documents
documents = []

# Process each PDF file
for file_name in pdf_files:
    if file_name.endswith('.pdf'):  # Ensure that the file is a PDF
        pdf_path = os.path.join(folder_path, file_name)  # Create the full path to the PDF
        loader = PyMuPDFLoader(pdf_path)  # Load the PDF
        doc = loader.load()  # Extract content
        documents.extend(doc)  # Add the content to the documents list

# Check the total number of documents (chunks of text)
print(f"Total documents extracted: {len(documents)}")




from langchain.text_splitter import CharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=100)

# Split documents in parallel using ThreadPolExecutor for faster processing
def split_in_batches(documents, batch_size=16):
    # Function to split a batch of documents
    split_results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(splitter.split_documents, documents[i:i+batch_size])
                   for i in range(0, len(documents), batch_size)]
        for future in futures:
            split_results.extend(future.result())
    return split_results

# Perform the split across all documents
split_documents = split_in_batches(documents, batch_size=16)




from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Initialize OpenAI embeddings (use your API key)
embeddings = OpenAIEmbeddings(api_key='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')
# Create FAISS index from the split documents
vectorstore = FAISS.from_documents(split_documents, embeddings)







import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os




import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image directory
image_dir = '/Users/navyakrishnabatchu/Desktop/Chatbot/chatbot-ui/Campus-Connect/chatbot-frontend/downloaded_images '

# Create a DataFrame with file paths and labels
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.png'))]
labels = ['class_1'] * len(image_paths)  # Replace 'class_1' with your actual labels if needed

# Create a DataFrame
df = pd.DataFrame({
    'filename': image_paths,
    'label': labels
})

# Define image size and batch size
img_height, img_width = 180, 180
batch_size = 32

# Use ImageDataGenerator with flow_from_dataframe
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Continue with model creation and training...


import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Define image directory
image_dir = '/Users/navyakrishnabatchu/Desktop/Chatbot/chatbot-ui/Campus-Connect/chatbot-frontend/downloaded_images '

# Create a DataFrame with file paths and labels (for this example, all images belong to 'class_1')
image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.jpg', '.png'))]
labels = ['class_1'] * len(image_paths)  # Modify the labels according to your actual class structure

# Create a DataFrame
df = pd.DataFrame({
    'filename': image_paths,
    'label': labels
})

# Define image size and batch size
img_height, img_width = 180, 180
batch_size = 32

# Set up ImageDataGenerator with a rescaling factor and validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

# Load the training data using flow_from_dataframe
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'binary' if only two classes
    subset='training'
)

# Load the validation data using flow_from_dataframe
validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    x_col='filename',
    y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'binary' if only two classes
    subset='validation'
)

# Build a CNN model based on the number of classes
if len(train_generator.class_indices) == 2:
    # Binary classification
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification with sigmoid
    ])

    # Compile the model for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
else:
    # Multi-class classification
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')  # Multi-class classification with softmax
    ])

    # Compile the model for multi-class classification
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print sample counts to verify
print(f"Number of training samples: {train_generator.samples}")
print(f"Number of validation samples: {validation_generator.samples}")

# Train the model
if validation_generator.samples > 0:
    model.fit(train_generator, validation_data=validation_generator, epochs=10)
else:
    model.fit(train_generator, epochs=10)

# Save the trained model for later use
model.save('professor_identification_cnn_model.h5')

# Function to preprocess the image
def preprocess_image(img_path, img_height, img_width):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.  # Normalize to [0, 1]
    return img_array

# Function to identify professor from the image
def identify_professor(img_path, model, train_generator):
    img_array = preprocess_image(img_path, img_height, img_width)
    predictions = model.predict(img_array)

    if len(train_generator.class_indices) == 2:
        # Binary classification case
        predicted_class = 1 if predictions[0] > 0.5 else 0
        class_labels = list(train_generator.class_indices.keys())
    else:
        # Multi-class classification case
        predicted_class = np.argmax(predictions, axis=1)
        class_labels = list(train_generator.class_indices.keys())

    predicted_label = class_labels[predicted_class[0]]
    return predicted_label

# Identify all downloaded images
def identify_all_images(image_dir, model, train_generator):
    for subdir, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):  # Adjust for image formats
                img_path = os.path.join(subdir, file)
                predicted_professor = identify_professor(img_path, model, train_generator)
                print(f"Image: {file} | Predicted Label: {predicted_professor}")

# Identify all images in the 'downloaded_images' directory
identify_all_images(image_dir, model, train_generator)



import os
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('professor_identification_cnn_model.h5')

# Image size (should match the size used during training)
img_height, img_width = 180, 180

# Function to preprocess the image
def preprocess_image(img_path, img_height, img_width):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Function to identify professor from the image
def identify_professor(img_path, model, train_generator):
    img_array = preprocess_image(img_path, img_height, img_width)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = list(train_generator.class_indices.keys())
    predicted_label = class_labels[predicted_class[0]]
    return predicted_label

# Function to display the image using Matplotlib
def display_image_with_label(img_path, predicted_label):
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Predicted Professor: {predicted_label}")
    plt.axis('off')  # Hide axes
    plt.show()

# Directory where the images are stored
image_dir = '/Users/navyakrishnabatchu/Desktop/Chatbot/chatbot-ui/Campus-Connect/chatbot-frontend/downloaded_images '

# Function to identify and display all images with predicted names
def identify_and_display_all_images(image_dir, model, train_generator):
    # Iterate through all subdirectories and images
    for subdir, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):  # Adjust for image formats
                img_path = os.path.join(subdir, file)
                predicted_professor = identify_professor(img_path, model, train_generator)
                print(f"Image: {file} | Predicted Professor: {predicted_professor}")

                # Display the image with the predicted label
                display_image_with_label(img_path, predicted_professor)

# Identify and display all images in the 'downloaded_images' directory
identify_and_display_all_images(image_dir, model, train_generator)



import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Function to display the image using OpenCV's imshow for non-Colab environments
def display_image_with_opencv(img_path):
    img = cv2.imread(img_path)
    if img is not None:
        cv2.imshow('Image', img)  # Display the image in a window
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close the window after a key press
    else:
        print(f"Could not load image: {img_path}")


# Load the trained CNN model
model = tf.keras.models.load_model('professor_identification_cnn_model.h5')

# Image size (should match the size used during training)
img_height, img_width = 180, 180

# Function to preprocess the image
def preprocess_image(img_path, img_height, img_width):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Function to identify professor from the image
def identify_professor(img_path, model, train_generator):
    img_array = preprocess_image(img_path, img_height, img_width)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = list(train_generator.class_indices.keys())
    predicted_label = class_labels[predicted_class[0]]
    return predicted_label



# Directory where the images are stored
image_dir = '/Users/navyakrishnabatchu/Desktop/Chatbot/chatbot-ui/Campus-Connect/chatbot-frontend/downloaded_images'

# Function to list all images
def list_all_images(image_dir):
    image_paths = []
    for subdir, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

# Function to find an image based on the professor's name (input from user)
def find_image_by_professor_name(professor_name, image_dir):
    for subdir, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')) and professor_name.lower() in file.lower():
                return os.path.join(subdir, file)
    return None

# Function to handle chatbot queries
def handle_query(user_input, model, train_generator):
    # Handle special command to list all images
    if 'list all images' in user_input.lower():
        images = list_all_images(image_dir)
        if images:
            for img in images:
                print(img)
        else:
            print("No images found.")
        return

    # Try to extract professor's name by looking for keywords like "image" or "professor"
    if 'image' in user_input.lower():
        professor_name = user_input.lower().replace('image', '').strip()
    elif 'professor' in user_input.lower():
        professor_name = user_input.lower().replace('professor', '').strip()
    else:
        professor_name = user_input.strip()  # Use entire input if no specific keyword found

    # Search for an image by professor's name
    img_path = find_image_by_professor_name(professor_name, image_dir)

    if img_path:
        # Predict the professor but do not print the predicted label
        predicted_professor = identify_professor(img_path, model, train_generator)

        # Display the image with OpenCV (no extra output)
        display_image_with_opencv(img_path)
        print(f"Displayed image for {professor_name}. You can continue.")
    else:
        print(f"No image found for {professor_name}")






import os

# Directory where images are stored
image_dir = '/Users/navyakrishnabatchu/Desktop/Chatbot/chatbot-ui/Campus-Connect/chatbot-frontend/downloaded_images '

# List all images in the directory
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(('.jpg', '.png')):
            print(os.path.join(root, file))



import openai
print(openai.__version__)
import openai
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from concurrent.futures import ThreadPoolExecutor
import urllib.parse  # To encode the location for Google Maps URL

# Initialize the OpenAI LLM with your API key
api_key = os.getenv('')
llm = OpenAI(api_key=api_key)

# Define your image directory here
image_dir = '/Users/navyakrishnabatchu/Desktop/Chatbot/chatbot-ui/Campus-Connect/chatbot-frontend/downloaded_images '

# Initialize Flask app
app = Flask(__name__)
CORS(app)

splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=100)

# Store the last query for editing
last_query = None

# Split documents in parallel using ThreadPoolExecutor for faster processing
def split_in_batches(documents, batch_size = 80):
    split_results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(splitter.split_documents, documents[i:i+batch_size])
                   for i in range(0, len(documents), batch_size)]
        for future in futures:
            split_results.extend(future.result())
    return split_results

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=api_key)

# Create FAISS index from the split documents
vectorstore = FAISS.from_documents(split_documents, embeddings)


# Initialize the retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

# Function to generate suggestions based on the current and previous query
def generate_suggestions(current_query, previous_responses):
    conversation_history = "\n".join(previous_responses)
    prompt = f"Based on the following conversation:\n\n{conversation_history}\n\nUser just asked: '{current_query}'. Suggest three relevant follow-up questions."

    # Generate suggestions using the LLM
    suggestions_response = llm(prompt)  # Use the OpenAI LLM to get suggestions
    # Split suggestions into a list (you can adjust based on your model's response format)
    suggestions = suggestions_response.strip().split('\n')
    return suggestions

# Function to handle text-based queries using LangChain and generate suggestions
def handle_text_query(user_input):
    chitchat_responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hey there! What can I do for you?",
        "Who are you": "I'm just a bot for Sacramento State University,  I'm here to help you",
        "How are you": "I am doing good, How about you?",
        "bye": "Goodbye! Feel free to reach out anytime.",
        "Campus Virtual tour link ": "https://www.youvisit.com/tour/60139/79933/",
        "How to Schedule Campus Tour appointment": "You can schedule the appointment using https://s.visitdays.com/csusao/ci/cjcskhncfv"
    }

    user_input_lower = user_input.lower()

    # Check if the input matches a casual conversation
    for key, value in chitchat_responses.items():
        if key in user_input_lower:
            return {"response": value, "suggestions": []}

    # If it's not chitchat, proceed with the regular query handling
    bot_response = qa_chain.run(user_input)

    # Generate follow-up query suggestions
    suggestions = generate_suggestions(user_input, [bot_response])

    return {"response": bot_response, "suggestions": suggestions}

# Function to handle image-based queries
def find_image_by_professor_name(professor_name, image_dir):
    for filename in os.listdir(image_dir):
        if professor_name.lower() in filename.lower():
            return os.path.join(image_dir, filename)
    return None

# Function to handle location-based queries and generate Google Maps link
def generate_google_maps_link(location_query):
    base_url = "https://www.google.com/maps/search/?api=1&query="
    location_encoded = urllib.parse.quote(location_query)
    full_url = f"{base_url}{location_encoded}"
    return full_url

# Flask API route for chat
@app.route('/chat', methods=['POST'])
def chat():
    global last_query
    data = request.json
    user_input = data.get('query', '')

    # Save the last query for editing later
    last_query = user_input

    # Handle image-based query
    if 'image' in user_input.lower() or 'professor' in user_input.lower():
        professor_name = user_input.lower().replace('image', '').replace('professor', '').strip()
        img_path = find_image_by_professor_name(professor_name, image_dir)

        if img_path:
            return jsonify({"response": f"/images/{os.path.basename(img_path)}", "type": "image", "suggestions": []})
        else:
            return jsonify({"response": f"No image found for {professor_name}", "type": "text", "suggestions": []})

    # Handle location-based queries (generate Google Maps link)
    if 'location' in user_input.lower() or 'map' in user_input.lower():
        location_query = user_input.lower().replace('location', '').replace('map', '').strip()
        google_maps_link = generate_google_maps_link(location_query)
        return jsonify({"response": f"Here is the Google Maps link: {google_maps_link}", "type": "text", "suggestions": []})

    # Handle text-based query using LangChain and generate suggestions
    result = handle_text_query(user_input)

    return jsonify({"response": result["response"], "suggestions": result["suggestions"], "type": "text"})

# Flask API to serve images
@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    img_path = os.path.join(image_dir, filename)
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return jsonify({"error": "Image not found"}), 404

# Route to retrieve the last query for editing
@app.route('/get_last_query', methods=['GET'])
def get_last_query():
    if last_query:
        return jsonify({"last_query": last_query})
    else:
        return jsonify({"last_query": "No previous query found."})

# Route to update and resubmit the last query
@app.route('/edit_and_resubmit', methods=['POST'])
def edit_and_resubmit():
    global last_query
    data = request.json
    new_query = data.get('new_query', '')

    # Update the last query
    if last_query:
        last_query = new_query
        # Re-run the updated query
        bot_response = handle_text_query(last_query)
        return jsonify({"response": bot_response, "type": "text"})
    else:
        return jsonify({"error": "No previous query to edit."})

# Feedback route
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_data = request.json
    response_text = feedback_data.get('response_text', '')
    feedback_value = feedback_data.get('feedback', 0)

    # Store the feedback (for now, just log it)
    print(f"Feedback received: {feedback_value} stars for response: {response_text}")

    # Respond with feedback acknowledgment
    return jsonify({"status": "Feedback received!"})

# Flask server for the chatbot
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)