import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np


class IEMOCAPDataset(Dataset):
    def __init__(self, train=True, filter_invalid=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('IEMOCAP_features/iemocap_multimodal_features.pkl', 'rb'), encoding='latin1')
        keys = list(self.trainVid if train else self.testVid)
        if filter_invalid:
            keys = [x for x in keys if self._is_valid_key(x)]
        self.keys = keys
        self.len = len(self.keys)

    def _is_valid_key(self, vid):
        if len(self.videoLabels[vid]) == 0:
            return False
        if len(self.videoText[vid]) == 0 or len(self.videoAudio[vid]) == 0 or len(self.videoVisual[vid]) == 0:
            return False
        if len(self.videoSpeakers[vid]) != len(self.videoText[vid]):
            return False
        return True

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(self.videoLabels[vid])),\
               torch.LongTensor(self.videoLabels[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path, train=True, filter_invalid=True):
        # MELD data structure: 
        # [videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoSentence, trainVid, testVid]
        # Note: MELD doesn't have visual features like IEMOCAP
        data = pickle.load(open(path, 'rb'))
        
        if len(data) == 9:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
                self.videoAudio, self.videoSentence, self.trainVid, \
                self.testVid, _ = data
        else:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
                self.videoAudio, self.videoSentence, self.trainVid, \
                self.testVid = data

        # For MELD, create synthetic visual features (zeros) since it doesn't have visual modality
        # This maintains compatibility with the model that expects 3 modalities
        self.videoVisual = {}
        for key in self.videoText.keys():
            text_len = len(self.videoText[key]) if isinstance(self.videoText[key], (list, np.ndarray)) else 1
            self.videoVisual[key] = np.zeros((text_len, 512), dtype=np.float32)

        keys = list(self.trainVid if train else self.testVid)
        if filter_invalid:
            keys = [x for x in keys if self._is_valid_key(x)]
        self.keys = keys
        self.len = len(self.keys)

    def _is_valid_key(self, vid):
        if len(self.videoLabels[vid]) == 0:
            return False
        if len(self.videoText[vid]) == 0 or len(self.videoAudio[vid]) == 0:
            return False
        if len(self.videoSpeakers[vid]) != len(self.videoText[vid]):
            return False
        return True

    def __getitem__(self, index):
        vid = self.keys[index]
        
        # Handle different data formats
        text_data = self.videoText[vid]
        if isinstance(text_data, (list, np.ndarray)):
            text_data = np.array(text_data, dtype=np.float32)
        
        audio_data = self.videoAudio[vid]
        if isinstance(audio_data, (list, np.ndarray)):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        visual_data = self.videoVisual[vid]  # Synthetic zeros for MELD
        if isinstance(visual_data, (list, np.ndarray)):
            visual_data = np.array(visual_data, dtype=np.float32)
        
        speakers_data = self.videoSpeakers[vid]
        if isinstance(speakers_data, (list, np.ndarray)):
            speakers_data = np.array(speakers_data, dtype=np.float32)
        
        return torch.FloatTensor(text_data), \
            torch.FloatTensor(visual_data), \
            torch.FloatTensor(audio_data), \
            torch.FloatTensor(speakers_data), \
            torch.FloatTensor([1] * len(self.videoLabels[vid])), \
            torch.LongTensor(self.videoLabels[vid]), \
            vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label += self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in
                dat]




