import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, length):
        self.encodings = encodings
        self.labels = labels
        self.length = length

    def _transform(self, encoding):
        start_idx = torch.randint(low=0, high=512, size=(1,))
        return {key:val[start_idx:start_idx+self.length] 
                for key, val in encoding.items()}

    def __getitem__(self, idx):
        item = {key: self.transform(torch.tensor(val[idx]))
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]) # check
        return item

    def __len__(self):
        return len(self.labels)