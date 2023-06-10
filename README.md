### Introduction to Speech-to-Text Transcription and Translation System

This report provides an overview of a Speech-to-Text Transcription and Translation system developed as part of an internship project. The system aims to convert spoken language into written text and facilitate translation to enable effective communication across language barriers. The project utilizes various technologies and frameworks such as Google Cloud Platform, natural language processing models, and database integration to achieve its objectives.

### Motivation

In today's globalized world, effective communication is crucial, and language barriers often pose significant challenges. Speech-to-Text Transcription and Translation systems have emerged as valuable tools to overcome these barriers and foster better understanding and collaboration among individuals speaking different languages. Such systems find applications in various domains, including customer support, healthcare, education, and international business.

The motivation behind developing this system was to create a user-friendly and efficient solution that enables real-time speech recognition, transcription, translation, and storage of the processed data. By automating the conversion of spoken language into written text and providing translations, the system aims to enhance accessibility, improve documentation, and facilitate multilingual communication.

## Objectives

The primary objectives of the Speech-to-Text Transcription and Translation system are as follows:

1. Real-Time Speech Recognition: The system captures audio input from a microphone and performs real-time speech recognition, converting the spoken language into text.
2. Language Translation: The system integrates with Google Cloud's translation services to provide translation capabilities, enabling the conversion of transcribed text into different target languages.
3. Natural Language Processing: The system utilizes natural language processing models, such as the T5 model, to generate informative answers to user queries based on the transcribed and translated text.
4. Database Integration: The system incorporates a MySQL database to store the transcriptions, translations, and user feedback, enabling data retrieval and analysis for further improvements.
5. User Interface: The system employs Streamlit, a user-friendly web application framework, to provide an intuitive interface for users to interact with the system, initiate recordings, view transcriptions, translations, and retrieve relevant information.

### System Architecture

The Speech-to-Text Transcription and Translation system architecture consists of several components:

1. Google Cloud Platform: The system leverages Google's Speech-to-Text API for real-time speech recognition and the Translation API for language translation. A Google service account is utilized to authenticate and access these APIs.
2. Natural Language Processing Models: The system utilizes the T5 model, trained on large-scale language understanding tasks, to generate informative answers based on user queries.
3. Database Integration: A MySQL database is integrated into the system to store transcriptions, translations, and user feedback. This enables efficient data management and retrieval for further analysis.
4. User Interface: Streamlit, a Python library for building web applications, is used to develop the user interface. It provides a seamless and interactive platform for users to initiate recordings, view transcriptions, translations, and obtain answers to their queries.

### Conclusion

The Speech-to-Text Transcription and Translation system developed as part of this internship project offers a comprehensive solution to overcome language barriers and facilitate effective communication. By combining real-time speech recognition, language translation, and natural language processing capabilities, the system enables users to transcribe spoken language into written text, translate it into various languages, and obtain informative answers to their queries.

The system's integration with Google Cloud Platform ensures accurate and efficient speech recognition and translation services, while the use of natural language processing models enhances the quality and relevance of generated answers. The inclusion of a database allows for seamless storage and retrieval of transcriptions, translations, and user feedback, facilitating further analysis and system improvements.

Overall, this Speech-to-Text Transcription and Translation system serves as a valuable tool for enabling multilingual communication, enhancing accessibility, and streamlining information

 exchange across language barriers. With ongoing advancements in speech recognition, machine translation, and natural language processing, such systems are expected to play a pivotal role in fostering global collaboration and understanding.

### Setting up a Google Cloud Account

You need a Google Service Account to run this script. Visit the link below to know how to set it up. Then download a credential json file. [Google Service Account](https://console.cloud.google.com/iam-admin/serviceaccounts) and [Get Credentials JSON File](https://developers.google.com/workspace/guides/create-credentials). Or you can follow the steps below to get the above task done:
1. Open the link (https://console.cloud.google.com/iam-admin/serviceaccounts) in your web browser.
2. Sign in to your Google account. If you don't have one, click on the "Create account" button to create a new Google account.
3. Once you are signed in, you will be redirected to the Google Cloud Console.
4. In the Cloud Console, click on the project dropdown menu at the top of the page and select the project for which you want to create a service account. If you don't have a project yet, you can create a new one by clicking on the "New Project" button and following the prompts.
5. In the left navigation menu, click on "IAM & Admin" to open the IAM & Admin page.
6. On the IAM & Admin page, click on "Service accounts" to view the list of existing service accounts for your project.
7. To create a new service account, click on the "Create service account" button.
8. In the "Create service account" dialog, enter a name for your service account in the "Service account name" field.
9. (Optional) Enter a description for your service account in the "Service account description" field.
10. Click on the "Create" button to create the service account.
11. On the next page, you can grant various roles and permissions to the service account. You can choose the appropriate roles based on your project's requirements. To grant a role, click on the "Add Member" button, enter the email address of the service account, select the role from the dropdown menu, and click on the "Save" button.
12. After assigning the roles, click on the "Done" button to finish creating the service account.
13. The service account will be listed on the "Service accounts" page. You can click on the service account name to view more details.
14. To generate a key for the service account, click on the vertical ellipsis (three dots) next to the service account name and select "Create key".
15. In the "Create key" dialog, choose the key type (JSON is recommended) and click on the "Create" button. This will generate a JSON key file that contains the credentials for the service account.
16. Download the JSON key file to your computer and securely store it. This key file will be used to authenticate your application or service with the Google APIs.

You have successfully set up a Google service account using the provided link. Make sure to keep the JSON key file safe and use it as needed for authentication in your applications or services.

Create a `.streamlit` folder and make a `secrets.toml` file in that folder. The downloaded json file will have all the details required to set up the `secrets.toml` file. 

Content of `secrets.toml` need to be like shown below:

```yml
# .streamlit/secrets.toml

mysql_host = ""
mysql_user = ""
mysql_password = ""
mysql_database = ""

[gcp_service_account]
type = "service_account"
project_id = "xxx"
private_key_id = "xxx"
private_key = "xxx"
client_email = "xxx"
client_id = "xxx"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "xxx"
```

### Packages and Libraries

The `requirements.txt` file specifies the dependencies required to run the files.

```
protobuf==3.19.5
PyAudio==0.2.13
six==1.16.0
streamlit==1.20.0
google-auth==2.16.2
google-cloud==0.34.0
google-cloud-speech==2.18.0
google-cloud-translate==3.11.0
pydub
mysql-connector-python
```

You can use this `requirements.txt` file to install the required dependencies using a package manager like `pip`. To install the dependencies, follow these steps:

1. Open a command-line interface or terminal.
2. Navigate to the directory containing the `requirements.txt` file.
3. Execute the following command:

```
pip install -r requirements.txt
```

This command will install all the dependencies listed in the `requirements.txt` file. Make sure you have Python and pip installed on your system before running the above command.

### For Linux Users

If someone is using Linux and needs to install additional packages to run the code, the following packages mentioned in the `packages.txt` file are required:

```
portaudio19-dev
python3-all-dev
ffmpeg
```

To install these packages on a Linux system, you can use the package manager specific to your Linux distribution. Here are the commands to install the packages on some popular Linux distributions:

**Ubuntu or Debian**:
```
sudo apt-get install portaudio19-dev python3-all-dev ffmpeg
```

**Fedora**:
```
sudo dnf install portaudio-devel python3-devel ffmpeg
```

**Arch Linux**:
```
sudo pacman -S portaudio python python-pip ffmpeg
```

Please note that package manager commands may vary based on the Linux distribution you are using. Make sure you have appropriate permissions or use `sudo` before the package manager command.

---


## `upload.py` 

### Introduction
`upload.py` is a script that performs audio transcription and translation using Google Cloud services. It allows users to upload an audio file and obtain the corresponding transcription, translation, and provide feedback. This documentation provides an overview of running the script, its dependencies, available functions, and usage instructions.

### Dependencies
The following dependencies are required to run `upload.py`:

- `google.oauth2`: OAuth2 client library for Google APIs
- `streamlit`: Web application framework for interactive UI
- `re`: Regular expression operations
- `sys`: System-specific parameters and functions
- `google.cloud.speech_v1p1beta1`: Google Cloud Speech-to-Text client library
- `io`: Core tools for working with streams
- `six`: Python 2 and 3 compatibility utilities
- `os`: Miscellaneous operating system interfaces
- `pydub`: Audio processing library
- `google.cloud.translate_v2`: Google Cloud Translation client library
- `tempfile`: Generate temporary files and directories
- `mysql.connector`: MySQL database connector

### Running the Script
To run the `upload.py` script, follow these steps:

1. Ensure that all the dependencies mentioned in the [Dependencies](#dependencies) section are installed.
2. Set up the required Google Cloud credentials. Refer to the documentation for obtaining and configuring the necessary credentials.
3. Open a command-line interface or terminal.
4. Navigate to the directory containing the `upload.py` file.
5. Execute the following command:

```python
streamlit run upload.py
```

6. The script will start running, and a web-based user interface will open in your default browser.

### Usage
Once the script is running, follow these steps to transcribe and translate an audio file:

1. Select the device microphone input type from the provided options: 'Laptop (Single Mic)', 'Laptop (Dual Mic)', 'Mobile (Single Mic)', 'Mobile (Dual Mic)'.
2. Choose the file type of the audio you want to upload: 'mp3' or 'ogg'.
3. Use the file uploader to select an audio file of the chosen file type.
4. The script will automatically transcribe the audio and display the transcription on the UI.
5. The transcription will be translated to the target language (Hindi) using Google Cloud Translation.
6. The translated text will be displayed on the UI.
7. Use the feedback slider to provide a feedback score between 0 and 100.
8. Click the "Upload Results to Database" button to store the transcription, translated text, and feedback score in a MySQL database.
9. The script will display the ID of the inserted transcription on the UI.

### Functions

#### connect_to_database()

This function establishes a connection to a MySQL database using the credentials provided in the script's secrets.

**Returns**
- `connection`: MySQL database connection object.

#### insert_transcription(transcript, translated_text, feedback)

This function inserts the provided transcription, translated text, and feedback into the connected MySQL database.

**Parameters**
- `transcript`: Transcription text obtained from audio.
- `translated_text`: Translated text of the transcription.
- `feedback`: Feedback score provided by the user.

**Returns**
- `transcription_id`: ID of the inserted transcription in the database.

#### main()

This function is the entry point of the script. It performs audio transcription, translation, and handles the user interface using the Streamlit framework.


---


## `realtime.py`

### Introduction
`realtime.py` is a script that performs speech-to-text transcription in real-time using Google Cloud's Speech-to-Text API. It allows users to transcribe audio from their microphone, obtain the transcription, translate it, and generate an answer to a related question using a pre-trained T5 model. This documentation provides an overview of running the script, its dependencies, available functions, and usage instructions.

### Dependencies
The following dependencies are required to run `realtime.py`:

- `google.oauth2`: OAuth2 client library for Google APIs
- `streamlit`: Web application framework for interactive UI
- `re`: Regular expression operations
- `sys`: System-specific parameters and functions
- `google.cloud.speech`: Google Cloud Speech-to-Text client library
- `pyaudio`: Python bindings for the PortAudio library
- `six`: Python 2 and 3 compatibility utilities
- `google.cloud.translate_v2`: Google Cloud Translation client library
- `mysql.connector`: MySQL database connector
- `transformers`: State-of-the-art natural language processing library (for T5 model)

### Running the Script
To run the `realtime.py` script, follow these steps:

1. Ensure that all the dependencies mentioned in the [Dependencies](#dependencies) section are installed.
2. Set up the required Google Cloud credentials. Refer to the documentation for obtaining and configuring the necessary credentials.
3. Open a command-line interface or terminal.
4. Navigate to the directory containing the `realtime.py` file.
5. Execute the following command:

```
streamlit run realtime.py
```

6. The script will start running, and a web-based user interface will open in your default browser.

### Usage
Once the script is running, follow these steps to transcribe audio from your microphone:

1. Click the "Start Recording" button to start transcribing audio from your microphone.
2. As you speak, the script will display the partial transcription in real-time.
3. When you finish speaking, click the "Stop Recording and Clear Text" button to stop the transcription process and clear the displayed text.
4. The script will then translate the transcribed text to the target language (Hindi) using Google Cloud Translation.
5. An answer to the transcribed text will be generated using a pre-trained T5 model.
6. The translated text, generated answer, and the original transcribed text will be stored in a MySQL database.
7. The ID of the inserted transcription will be displayed on the UI.

### Functions

#### answer_question(question)

This function takes a question as input and generates an answer using a pre-trained T5 model.

**Parameters**
- `question`: The question to generate an answer for.

**Returns**
- `answer`: The generated answer for the question.

#### connect_to_database()

This function establishes a connection to a MySQL database using the credentials provided in the script.

**Returns**
- `connection`: MySQL database connection object.

#### insert_transcription(transcript, translated_text, feedback=90)

This function inserts the transcribed text, translated text, and feedback score into a MySQL database.

**Parameters**
- `transcript`: The transcribed text.
- `translated_text`: The translated text of the transcription.
- `feedback`: Feedback score provided by the user (default is 90).

**Returns**
- `transcription_id`: ID of the inserted transcription in the database.

#### MicrophoneStream
A class that represents a microphone audio stream. It provides methods to start and stop recording audio from the microphone and generates audio chunks.

#### listen_print_loop(responses)

This function processes the audio responses received from the Google Cloud Speech-to-Text API. It displays the partial transcriptions in real-time and performs translation, answer generation, and database insertion.

**Parameters**
- `responses`: Audio responses received from the Google Cloud Speech-to-Text API.

#### main()

This function is the entry point of the script. It sets up the necessary configurations for audio transcription, handles the user interface using Streamlit, and manages the recording and transcription process.

---


## `dashboard.py`

## Introduction
`dashboard.py` is a script that allows users to upload data to a MySQL database using a web-based dashboard created with Streamlit. It provides a user interface to enter data type, title, and content for multiple entries and inserts them into the MySQL database. This documentation provides an overview of running the script, its dependencies, available functions, and usage instructions.

### Dependencies
The following dependencies are required to run `dashboard.py`:

- `streamlit`: Web application framework for interactive UI
- `mysql.connector`: MySQL database connector
- `google.oauth2`: OAuth2 client library for Google APIs

### Running the Script
To run the `dashboard.py` script, follow these steps:

1. Ensure that all the dependencies mentioned in the [Dependencies](#dependencies) section are installed.
2. Set up the required Google Cloud credentials. Refer to the documentation for obtaining and configuring the necessary credentials.
3. Open a command-line interface or terminal.
4. Navigate to the directory containing the `dashboard.py` file.
5. Execute the following command:

```
streamlit run dashboard.py
```

6. The script will start running, and a web-based user interface will open in your default browser.

### Usage
Once the script is running, follow these steps to upload data to the MySQL database:

1. Select the data type from the dropdown list. The available options are "Question & Answer," "Blog," and "Forum."
2. Enter the number of entries to add in the "Enter the number of entries to add" field.
3. For each entry, enter a title in the corresponding "Enter title" text field and enter the content in the corresponding "Enter content" text area.
4. Click the "Submit" button to upload the entered data to the MySQL database.
5. The script will establish a connection to the database using the configured credentials.
6. The entered data will be inserted into the `content` table in the database, with the specified data type, title, and content.
7. Once the data is successfully uploaded, a success message will be displayed on the UI.
8. The database connection will be closed.

### Functions

#### connect_to_db()

This function establishes a connection to the MySQL database using the configured credentials.

**Returns**
- `connection`: MySQL database connection object.

#### insert_data(connection, data_type, title, content)

This function inserts the data type, title, and content into the MySQL database.

**Parameters**
- `connection`: MySQL database connection object.
- `data_type`: The type of data to be inserted (e.g., "Question & Answer", "Blog", "Forum").
- `title`: The title of the entry.
- `content`: The content of the entry.

#### main()

This function is the entry point of the script. It sets up the Streamlit user interface, handles user inputs, and manages the database insertion process.

---

## `model/train.py`

```python
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering, TrainingArguments, Trainer
import tensorflow as tf

# Load the dataset from a CSV file
data = pd.read_csv('data.csv')

# Create a Hugging Face Dataset from the loaded data
dataset = Dataset.from_pandas(data)

# Split the dataset into training and testing sets
dataset = dataset.train_test_split(test_size=0.1)

# Initialize the tokenizer with the BERT-base-uncased model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function to tokenize the questions and answers
def tokenize(batch):
    return tokenizer(batch['questions'], batch['answers'], padding='max_length', truncation=True)

# Tokenize the training dataset
train_dataset = dataset['train'].map(tokenize, batched=True, batch_size=len(dataset['train']))

# Tokenize the validation dataset
val_dataset = dataset['test'].map(tokenize, batched=True, batch_size=len(dataset['test']))

# Set the format of the datasets to be compatible with TensorFlow
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize the model with the BERT-base-uncased model architecture
model = TFAutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

# Configure the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer with the model, training arguments, and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Function to get the answer for a given question
def get_answer(question):
    # Tokenize the input question
    inputs = tokenizer(question, return_tensors='tf')
    
    # Pass the inputs through the model to get answer scores
    outputs = model(inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Get the indices of the start and end positions of the answer
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0] 
    answer_end = (tf.argmax(answer_end_scores, axis=1) + 1).numpy()[0] 

    # Convert the token IDs to the actual answer string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

    return answer
```

This script performs question answering using the BERT-base-uncased model. Here's an overview of its functionality:

1. The script starts by importing the required libraries and modules.
2. The dataset is loaded from a CSV file using `pd.read_csv` and converted into a Hugging Face `Dataset` object.
3. The dataset is split into training and testing sets using the `train_test_split` method.
4. The BERT-base-uncased tokenizer is initialized using `AutoTokenizer.from_pretrained`.
5. A `tokenize` function is defined to tokenize the questions and answers in the dataset.
6. The training and validation datasets are tokenized using the `map` method.
7. The format of the datasets is set to be compatible with TensorFlow using the `set_format` method.
8. The BERT-base-uncased model is initialized using `TFAutoModelForQuestionAnswering.from_pretrained`.
9. The training arguments are configured using `TrainingArguments`.
10. A `Trainer` object is initialized with the model, training arguments, and datasets.
11. The model is trained using the `train` method of the `Trainer` object.
12. A `get_answer` function is defined to get the answer for a given question.
13. In the `get_answer` function, the input question is tokenized, and the model is used to obtain answer scores.
14. The indices of the start and end positions of the answer are obtained using `argmax`.
15. The token IDs are converted to the actual answer string using the tokenizer.

This script can be used to train a question answering model using BERT and obtain answers for new questions.
