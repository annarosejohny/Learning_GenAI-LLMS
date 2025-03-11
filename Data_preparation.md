## Data preparation

### Tokenization
- Breaking a sentence into smaller words or tokens
- Tokenizer- breaks text into individual tokens
- Tokenizers, such as NLTK, SpaCy generate tokens
- Tokenization methods
  - Word-based: divides text into individual words
    - advantage: preserves semantic meaning
    - disadvantage: increase the models overall vocabulary
  - charachter-based: splits text into individual characteristics
    - advantage: smaller vocabularies
    - disadvantage: may not convey the same information as entire words. Increase input dimensionality and computational needs 
  - subword-based: frequently use words unsplit; infrequent words broken down
    - advantage: combines advantages of word-based and character-based tokenization
    ![sub_word_based](https://github.com/user-attachments/assets/39e78d93-6792-48e1-abf5-a48d3cc76c4a)
- tokenization and indexing in pytorch
  - Uses **torchtext** library for tokenization
  - Uses the **build_vocab_from_iterator** function
    - creates vocabulary from tokens
    - assigns each token a unique index
![dataset](https://github.com/user-attachments/assets/ada7f1a5-95de-4e94-9800-923c133cd4ef)
![tokenizer](https://github.com/user-attachments/assets/07dd1ff5-7265-4b9e-ab61-4ebeca1755b2)
![tokenizer2](https://github.com/user-attachments/assets/fa13c51c-3ede-4e23-a8c6-8a1246ebaebe)
![tokenizer3](https://github.com/user-attachments/assets/15f971dc-b8a6-42de-bfb0-7449ee411bd0)
- Applying **vocab** function gives the indices of those tokens
![vocab](https://github.com/user-attachments/assets/387c4a33-4b99-45a8-82c2-45c7b7f94841)
- creating tokens and indices
  ![tokenizer4](https://github.com/user-attachments/assets/4a25a850-7018-431e-9656-b5322fa9edac)
- Special tokens and build vocab from iterator: special tokens can be made by addinhg "**<bos>**"(beginning of sentence) and "**<eos>**"(end of senetnce) to starting and ending and then repeat the same using loop. 
![tokenizer5](https://github.com/user-attachments/assets/e847a1fa-4cc9-4e3e-a40b-37c8cb736c5d)
- Also pad tokenizer using **<pad>** to match the length of sentences. 
![tokenizer6](https://github.com/user-attachments/assets/43426bee-3851-40f1-9d2d-6bfc5c63f6f2)
- "**<unk>**":stands for "unknown" and represents words that were not seen during vocabulary building, usually during inference on new text.
- Eg python code
  **new_line = "I learned about embeddings and attention mechanisms."**
  **tokenized_new_line = tokenizer_en(new_line)**
  **tokenized_new_line = ['<bos>'] + tokenized_new_line + ['<eos>']**
  **new_line_padded = tokenized_new_line + ['<pad>'] * (max_length -len(tokenized_new_line))**
  **new_line_ids = [vocab[token] if token in vocab else vocab['<unk>'] for token in new_line_padded]**
  **print("Token IDs for new line:", new_line_ids)**

  
- Libraries used for tokenization
  - **nltk** or natural language toolkit, will be employed for data management tasks. It offers comprehensive tools and resources for processing natural language text, making it a valuable choice for tasks such as text preprocessing and analysis.

  - **spaCy** is an open-source software library for advanced natural language processing in Python. spaCy is renowned for its speed and accuracy in processing large volumes of text data.

 - **BertTokenizer** is part of the Hugging Face Transformers library, a popular library for working with state-of-the-art pre-trained language models. BertTokenizer is specifically designed for tokenizing text according to the BERT model's specifications.

 - **XLNetTokenizer** is another component of the Hugging Face Transformers library. It is tailored for tokenizing text in alignment with the XLNet model's requirements.

 - **torchtext** It is part of the PyTorch ecosystem, to handle various natural language processing tasks. It simplifies the process of working with text data and provides functionalities for data preprocessing, tokenization, vocabulary management, and batching.


## Data loaders
- Efficient loading and preprocessing of textual data
- Memory optimization through on-the-fly preprocessing
- Seamless integration with pytorch training pipeline
- simplified data augmentation and preprocessing
- efficient batching and shuffling of data
- Dataset - Collection of data samples and their labels
  - Training set
  - validation set
  - test set
- Custom dataset
![custom_dataset](https://github.com/user-attachments/assets/bf5caaa9-42f3-40fa-9bbc-3031deddc0e4)
- **Dataloader**
  - output data as batches instead of one sample at a time.
  - It is an iterator used for loading, shuffling, and batching dataset
- **Iterator** -
  - object that can be looped over
  - two methods: iter() and next()
  - data_iter = iter(dataloader) - loads the dataset
  - next(data_iter) - load next item in the dataset
- Collate function
  - Data tranformation
  - converting tokenized indices
  - tokenization
  - transforming the result into a tensor
    ![collate_Fn](https://github.com/user-attachments/assets/98dcb312-813c-4570-a8ed-ac53792e9d00)


## Data quality
Data quality refers to the accuracy, consistency, and completeness of the dataset used for training. Poor-quality data introduces noise that can lower a model's accuracy and reliability. 
Here are practices to ensure high data quality:

- **Noise reduction**: Remove irrelevant or repetitive data to help the model focus on significant patterns and linguistic structures. For example, clean datasets by removing typos and irrelevant information, such as forum tags, to enhance quality.
- **Consistency checks**: Regularly verify consistency to prevent conflicting or outdated information from confusing the model. Consistency is essential for entities, like public figure names or technical terms, ensuring uniform usage throughout the dataset.
- **Labeling quality**: For labeled datasets, accurate labeling is crucial to avoid misleading the model. Clear guidelines for human annotators can reduce subjective errors and improve labeling quality.

## Diverse representation
A diverse dataset enhances a model’s inclusivity, ensuring it responds accurately to varied cultural, demographic, and regional inputs. Without diverse representation, models may reflect narrow views, leading to unintentional biases. 
Here’s how to achieve meaningful diversity:
- **Inclusion of varied demographics**: Incorporate text from various demographic groups to avoid over-representing a single perspective. Include sources in multiple languages or dialects and represent diverse cultural norms to improve global applicability.
- **Balanced data sources**: Draw a balanced dataset from sources such as news, social media, literature, and technical documents. This broadens the model’s knowledge base and reduces dependence on any single source.
- **Regional and linguistic variety**: Including datasets from diverse regions and languages expands the model’s linguistic and cultural context, enhancing accuracy in multilingual contexts and better supporting translation tasks.


























