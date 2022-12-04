import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms


img_model_file = 'models/nfnet.pt'
text_model_file = 'models/bert.pt'
bert_path = '../data/bert-indo-15g'
database = '../data/train.csv'
img_dir = 'train_images/'

class ImageHandler():
    def __init__(self, size=224):
        super().__init__()
        self.transform =  transforms.Compose([
                                transforms.Resize(size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                            ])
        self.model = torch.jit.load(img_model_file)
    def open_img(self, path):
        return Image.open(path)


    def preprocess(self,image):
        return self.transform(image).unsqueeze(0)
    
    def run(self, input):
        image = self.open_img(input)
        processed = self.preprocess(image)
        ids, sim = self.model(processed)
        return ids.numpy()


class TextHandler():
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model = torch.jit.load(text_model_file)
    
    @staticmethod
    def string_escape(s, encoding='utf-8'):
        return s.encode('latin1').decode('unicode-escape').encode('latin1').decode(encoding)               

    def preprocess(self, text):
        text= TextHandler.string_escape(text)
        encodings = self.tokenizer(text, padding = 'max_length', max_length=100, truncation=True,return_tensors='pt')
        keys =['input_ids', 'attention_mask']
        return tuple(encodings[key] for key in keys)
    
    def run(self, input):
        processed = self.preprocess(input)
        ids, sim = self.model(*processed)
        return ids.numpy()

class DB():
    def __init__(self):
        super().__init__()
        self.db = pd.read_csv(database)
    
    def query(self,ids):
   
        res = self.db[['title','image']].iloc[ids]
        # res['path'] = img_dir + res['image'].astype(str)
        return res['image'].values.tolist() , res['title'].values.tolist()
        # return res['title'].to_list()

def combine_ids(ids_1, ids_2):
    return np.intersect1d(ids_1,ids_2)



