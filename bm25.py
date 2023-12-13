import pickle
from rank_bm25 import BM25Okapi
import json
from tqdm import tqdm

def train_bm25(passages):
    tokenized_passages = [passage['Description'].split() for passage in tqdm(passages)]
    bm25 = BM25Okapi(tokenized_passages)
    return bm25

def save_bm25_model(bm25_model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(bm25_model, file)

def load_bm25_model(filename):
    with open(filename, 'rb') as file:
        bm25_model = pickle.load(file)
    return bm25_model

print('---'*20 + 'Loading Data' + '---'*20)
# Dữ liệu mẫu
passages = json.load(open('/home/luungoc/BTL-2023.1/Deep learning/Data Json/collection.json', 'r'))

print('---'*20 + 'Training BM25' + '---'*20)
# Huấn luyện mô hình
bm25_model = train_bm25(passages)

print('---'*20 + 'Saving Model' + '---'*20)
# Lưu mô hình
save_bm25_model(bm25_model, './BM25/bm25_model.pkl')


print('---'*20 + 'Loading BM25' + '---'*20)
# Load mô hình từ file
loaded_bm25_model = load_bm25_model('./BM25/bm25_model.pkl')

print('---'*20 + 'Query Data' + '---'*20)
# Sử dụng mô hình đã load để đánh giá truy vấn
query = "your_query_here"
scores = loaded_bm25_model.get_scores(query.split())

# Xếp hạng các đoạn văn theo điểm số giảm dần
ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

# In ra kết quả
for passage, score in ranked_passages:
    print(f"Passage: {passage}, Score: {score}")
