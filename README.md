# Sentiment Analysis using DistilBERT

A sentiment analysis project using a fine-tuned DistilBERT model on the IMDB dataset to classify movie reviews as **positive** or **negative**.  
The project includes training, evaluation, and a simple inference app built with Streamlit.

## Demo

![image](https://github.com/user-attachments/assets/f9150a93-c774-4681-a5a0-2bab87bac845)


---

## Project Structure
```bash
.
├── app/                 
├── datasets/
│   └── imdb/
├── models/              
├── notebooks/           
├── results/             
├── .gitattributes       
├── LICENSE              
├── README.md            
└── requirements.txt
```

---   

## Features

- Binary classification (Positive/Negative)
- Fine-tuned DistilBERT (lighter & faster BERT variant)
- Model trained on IMDB reviews dataset
- Interactive web app for inference (Streamlit)
- Visualizations and evaluation metrics

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/Prabhnoor4/sentiment-analysis-using-distilbert.git
cd sentiment-analysis-using-distilbert
```
---
## Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Install dependencies
```bash
pip install -r requirements.txt
```
---
## Training the Model
### Navigate to the notebooks/ directory and open the training notebook.

```bash
cd notebooks
```

## Running the Streamlit App

```bash
cd app
streamlit run app.py
```

---

## License
### This project is licensed under the MIT License — see the LICENSE file for details.



