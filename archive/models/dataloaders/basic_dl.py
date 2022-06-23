import torch
import pickle
from sentence_transformers import SentenceTransformer, util

class RedditDatasetFromPickle(torch.utils.data.Dataset):
    def __init__(self, file_loc, remove_last=False, device='cuda:0'):
        self.training_data = pickle.load(open(file_loc, 'rb'))
        self.sbert_model = SentenceTransformer('msmarco-distilbert-cos-v5')
        self.remove_last = remove_last
        self.device = device
    def __len__(self):
        return len(self.training_data)
    def __getitem__(self, i):
        encoded_context = None
        if self.remove_last:
            encoded_context = self.sbert_model.encode(self.training_data[i]['full_context'][:-1],convert_to_tensor=True)
        else:
            encoded_context = self.sbert_model.encode(self.training_data[i]['full_context'],convert_to_tensor=True)
        
        Y = self.sbert_model.encode(self.training_data[i]['text'], convert_to_tensor=True)
        
        return (encoded_context.to(self.device), Y.to(self.device))
