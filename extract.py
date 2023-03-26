import fitz
import re
from sentence_transformers import SentenceTransformer, util
import json
import openai
from openai.embeddings_utils import get_embedding, get_embeddings, cosine_similarity
import pickle


def read_open_sk_key():
    with open('openai.sk') as f:
        return f.read().strip()
    

openai.api_key = read_open_sk_key()


def create_index_openai(texts):
    just_texts = list(texts.keys())
    embeddings = get_embeddings(just_texts, engine='text-embedding-ada-002')
    for i, embedding in enumerate(embeddings):
        texts[just_texts[i]] = {'embedding': embedding, 'page': texts[just_texts[i]]}
    with open("openai_index.json", "w") as f:
        json.dump(texts, f)
    

def load_openai_index():
    with open("openai_index.json", "r") as f:
        return json.load(f)
    

def load_sentence_transformer_index():
    with open("sentence_transformer_index.pickle", "rb") as f:
        return pickle.load(f)


def create_index_sentence_transformer(texts, model):
    just_texts = list(texts.keys())
    embeddings = embed_texts(texts, model)
    for i, embedding in enumerate(embeddings):
        texts[just_texts[i]] = {'embedding': list(embedding), 'page': texts[just_texts[i]]}
    with open("sentence_transformer_index.pickle", "wb") as f:
        pickle.dump(texts, f)


def get_top_similar_openai(question, index, topN=10):
    question_embedding = get_embedding(question, engine='text-embedding-ada-002')
    scores = []
    for i, text in enumerate(index):
        score = cosine_similarity(question_embedding, index[text]['embedding'])
        scores.append((i, score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:topN]
    return scores


def extract_blocks_texts_from_pdf(path):
    doc = fitz.open(path)
    texts = {} # text -> [(text_block, page_index)]
    for page_index, page in enumerate(doc):
        blocks = page.get_text('blocks')
        for block in blocks:
            if block[-1] == 0:
                text = block[4].replace('\n', ' ').replace('\r', ' ').strip()
                text = re.sub(r'\s+', ' ', text)
                block_index = block[5]
                try:
                    texts[text].append((block_index, page_index))
                except:
                    texts[text] = [(block_index, page_index)]
    return texts


def cleanse_text(text):
    return text.encode(encoding='ASCII', errors='ignore').decode()


def extract_texts_from_pdf(path):
    doc = fitz.open(path)
    texts = {} # text -> page_index
    for page_index, page in enumerate(doc):
        text = page.get_text()
        text = text.replace('\n', ' ').replace('\r', ' ').strip()
        text = re.sub(r'\s+', ' ', text)
        text = cleanse_text(text)
        if text:
            texts[text] = page_index
    return texts


def load_sentence_model(model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1'):
    model = SentenceTransformer(model_name)
    return model


def embed_texts(texts, sentence_model):
    embeddings = sentence_model.encode(list(texts.keys()))
    return embeddings


def find_similar_texts_for_question(question, texts_embeddings, model, topN=10):
    question_embedding = model.encode(question)
    
    scores = util.dot_score(question_embedding, texts_embeddings)[0].cpu().tolist()
    doc_index_score_pairs = list(zip(list(range(len(texts_embeddings))), scores))

    #Sort by decreasing score
    doc_index_score_pairs = sorted(doc_index_score_pairs, key=lambda x: x[1], reverse=True)[:topN]
    
    return doc_index_score_pairs


def ask_me_a_question(question, texts, model):
    texts_embeddings = embed_texts(texts, model)
    doc_index_score_pairs = find_similar_texts_for_question(question, texts_embeddings, model)
    for entry in doc_index_score_pairs:
        doc_index, score = entry[0], entry[1]
        text = list(texts.keys())[doc_index]
        print(f'{score:.4f} {text}')


def main_blocks():
    model = load_sentence_model()
    texts = extract_blocks_texts_from_pdf("manual_c3.pdf")
    ask_me_a_question("How to activate autopilot?", texts, model)
    

def main_full_text():
    model = load_sentence_model()
    texts = extract_texts_from_pdf("manual_c3.pdf")
    ask_me_a_question("How to activate autopilot?", texts, model)
    

def main_create_openai_index():
    texts = extract_texts_from_pdf("manual_c3.pdf")
    create_index_openai(texts)


def main_create_sentence_tranformer_index():
    model = load_sentence_model()
    texts = extract_texts_from_pdf("manual_c3.pdf")
    create_index_sentence_transformer(texts, model)

def main_ask_question_openai():
    index = load_openai_index()
    question = "How to activate autopilot?"
    top_index_scores = get_top_similar_openai(question, index, topN=10)
    for i, score in top_index_scores:
        text = list(index.keys())[i]
        print(f'{score:.4f} {index[text]["page"]} {text}')
    

def main_ask_chatgpt_a_question():
    index_openai = load_openai_index()
    index_st = load_sentence_transformer_index()
    st_model = load_sentence_model()
    question = "How to change a flat tire?"
    top_openai_index_scores = get_top_similar_openai(question, index_openai, topN=5)
    texts_embeddings = [entry['embedding'] for entry in index_st.values()]
    top_st_index_scores = find_similar_texts_for_question(question, texts_embeddings, st_model, topN=5)
    
    open_ai_pages = set([index_openai[list(index_openai.keys())[i]]['page'] for i, score in top_openai_index_scores])
    
    
    prompt = f"""You are trying to help an user with a question regarding their car. Here is the question: {question}
    
    Here is a list of the texts from potential helping pages from the car's manual:
    
    
    """
    counter = 1
    for i, score in top_openai_index_scores:
        text = list(index_openai.keys())[i]
        prompt += f"{counter}. {text}\n\n"
        counter += 1
    
    for i, score in top_st_index_scores:
        text = list(index_st.keys())[i]
        if index_st[text]['page'] in open_ai_pages:
            continue
        prompt += f"{counter}. {text}\n\n"
        counter += 1
        
    print(prompt)
    print()
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": prompt},
            ]
        )
    print(response['choices'][0]['message']['content'])


if __name__ == "__main__":
    # main_full_text()
    # main_create_openai_index()
    # main_ask_question_openai()
    # main_create_sentence_tranformer_index()
    main_ask_chatgpt_a_question()