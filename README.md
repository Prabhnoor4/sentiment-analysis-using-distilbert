Sentiment Analysis using DistilBERT
A sentiment analysis project leveraging DistilBERT to classify movie reviews from the IMDB dataset as positive or negative.
This repository covers everything from data preprocessing and model training to evaluation and deployment.

🚀 Features
Fine-tuned DistilBERT model on the IMDB movie review dataset.

Classifies text sentiment as positive or negative.

Complete pipeline: datasets → model training → evaluation → app deployment.

Clean Jupyter notebooks to understand and replicate the workflow.

🛠️ Tech Stack
Python

Hugging Face Transformers

PyTorch / TensorFlow

scikit-learn

Streamlit (for app deployment — if used)

📂 Project Structure
graphql
Copy
Edit
.
├── app/                 # Code for running the inference app (optional: Streamlit, Flask, etc.)
├── datasets/imdb/       # IMDB dataset (processed data)
├── models/              # Fine-tuned DistilBERT model and tokenizer
├── notebooks/           # Jupyter notebooks (EDA, training, evaluation)
├── results/             # Plots, evaluation metrics, results
├── .gitattributes       # Git LFS tracking (for large model files)
├── LICENSE              # License information
├── README.md            # Project documentation
└── requirements.txt     # Required Python libraries
⚙️ Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/Prabhnoor4/sentiment-analysis-using-distilbert.git
cd sentiment-analysis-using-distilbert
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
(Optional) Download dataset and model files
Make sure the IMDB dataset and trained model files are available in the correct folders (datasets/ and models/).

💡 Usage
Run Sentiment Analysis in Jupyter Notebook
You can open and run the notebooks in the notebooks/ folder to train or evaluate the model.

Run Inference App
If an app (Streamlit or Flask) is set up in the app/ folder, run:

bash
Copy
Edit
cd app
python app.py
Or if it's Streamlit:

bash
Copy
Edit
streamlit run app.py
📈 Results
The fine-tuned DistilBERT model achieves strong performance on the IMDB dataset, with high accuracy and fast inference times.

🙌 Acknowledgements
Hugging Face Transformers

IMDB Dataset

DistilBERT paper


