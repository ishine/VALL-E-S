import logging
import librosa
import math
import numpy as np
from torch.utils.data import Dataset
from dataset.labels import Labeller
from dataset.utils import get_duration_sec, load_audio
from dataset.audio_processor import AudioTokenizer, tokenize_audio


class FilesAudioDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.sr = cfg.dataset.sr
        self.channels = cfg.dataset.channels
        self.min_duration = cfg.dataset.min_duration or math.ceil(cfg.dataet.sample_length / cfg.dataset.sr)
        self.max_duration = cfg.dataset.max_duration or math.inf
        self.sample_length = cfg.dataset.sample_length
        assert cfg.dataset.sample_length / cfg.dataset.sr < self.min_duration, f'Sample length {cfg.dataset.sample_length} per sr {cfg.dataset.sr} ({cfg.dataset.sample_length / cfg.dataset.sr:.2f}) should be shorter than min duration {self.min_duration}'
        self.aug_shift = cfg.aug_shift
        self.labels = cfg.labels
        self.device = cfg.device
        self.init_dataset(cfg)

    def filter(self, files, durations):
        # Remove files too short or too long
        keep = []
        for i in range(len(files)):
            if durations[i] / self.sr < self.min_duration:
                continue
            if durations[i] / self.sr >= self.max_duration:
                continue
            keep.append(i)
        logging.info(f'self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}')
        logging.info(f"Keeping {len(keep)} of {len(files)} files")
        self.files = [files[i] for i in keep]
        self.durations = [int(durations[i]) for i in keep] #サンプル長
        self.cumsum = np.cumsum(self.durations) #サンプル長の累積和

    def init_dataset(self, cfg):
        # Load list of files and starts/durations
        files = librosa.util.find_files(f'{cfg.dataset.audio_files_dir}', ['mp3', 'opus', 'm4a', 'aac', 'wav'])
        logging.info(f"Found {len(files)} files. Getting durations")
        cache = cfg.exp_dir
        durations = np.array([get_duration_sec(file, cache=cache) * self.sr for file in files])  # Could be approximate
        self.filter(files, durations)

        if self.labels:
            self.labeller = Labeller(cfg.dataset.max_bow_genre_size, cfg.dataset.n_tokens, self.sample_length, v3=cfg.dataset.labels_v3)

    def get_index_offset(self, item):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length//2
        shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
        offset = item * self.sample_length + shift # Note we centred shifts, so adding now
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f'Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}'
        index = np.searchsorted(self.cumsum, midpoint)  # index <-> midpoint of interval lies in this song
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index] # start and end of current song
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
        if offset > end - self.sample_length: # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        elif offset < start: # Going under song
            offset = min(end - self.sample_length, offset + half_interval)  # Now should fit
        assert start <= offset <= end - self.sample_length, f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"
        offset = offset - start
        return index, offset

    def get_metadata(self, filename, test):
        """
        Insert metadata loading code for your dataset here.
        If artist/genre labels are different from provided artist/genre lists,
        update labeller accordingly.

        Returns:
            (artist, genre, full_lyrics) of type (str, str, str). For
            example, ("unknown", "classical", "") could be a metadata for a
            piano piece.
        """
        
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        ls = [l.split("|") for l in lines]
        ls_T = list(zip(*ls))
        
        artist, genre, lang, lyrics, gender, pitch, speed, volume, emotion =\
            list(ls_T[0]), list(ls_T[1]), list(ls_T[2]), list(ls_T[3]), list(ls_T[4]),\
                list(ls_T[5]), list(ls_T[6]), list(ls_T[7]), list(ls_T[8])
        
        return artist[0], genre[0], lang[0], lyrics[0],gender[0],\
            pitch[0], speed[0],volume[0], emotion[0]

    def get_song_chunk(self, index, offset, test=False):
        filename, total_length = self.files[index], self.durations[index]
        data, sr = load_audio(filename, sr=self.sr, offset=offset, duration=self.sample_length)
        assert data.shape == (self.channels, self.sample_length), f'Expected {(self.channels, self.sample_length)}, got {data.shape}'
        codec = AudioTokenizer(self.device)
        if data.size(-1) / sr > 15:
            raise ValueError(f"Prompt too long, expect length below 15 seconds, got {data / sr} seconds.")
        if data.size(0) == 2:
            data = data.mean(0, keepdim=True)
        _, codes = tokenize_audio(codec, (data, sr))
        audio_tokens = codes.transpose(2,1).cpu().numpy()    
        
        if self.labels:
            ann_filename = filename.replace(".wav", ".txt")
            phoneme_duration_filename = filename.replace(".wav", ".TextGrid")
            artist, genre, lang, lyrics = self.get_metadata(ann_filename, test)
            labels = self.labeller.get_label(artist, genre, lang, lyrics, total_length, offset)
            labels['audio_features'] = audio_tokens
            labels['audio_features_lens'] = audio_tokens.shape[1]
            labels['audio'] = data
            return labels
        else:
            return data.T
    
    def get_dur(self, idx):
        return self.durations[idx]

    def get_item(self, item, test=False):
        index, offset = self.get_index_offset(item)
        return self.get_song_chunk(index, offset, test)

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, item):
        return self.get_item(item)
