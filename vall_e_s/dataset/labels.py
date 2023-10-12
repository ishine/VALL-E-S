import torch 
import numpy as np
from dataset.artist_genre_processor import ArtistGenreProcessor
from dataset.text_processor import TextProcessor, get_text_token_collater
text_collater = get_text_token_collater()

language_dict = {
    'en': 0,
    'zh': 1,
    'ja': 2,
}


def get_relevant_lyric_tokens(full_tokens, n_tokens, total_length, offset, duration):
    if len(full_tokens) < n_tokens:
        tokens = [0] * (n_tokens - len(full_tokens)) + full_tokens
        indices = [-1] * (n_tokens - len(full_tokens)) + list(range(0, len(full_tokens)))
    else:
        assert 0 <= offset < total_length
        midpoint = int(len(full_tokens) * (offset + duration / 2.0) / total_length)
        midpoint = min(max(midpoint, n_tokens // 2), len(full_tokens) - n_tokens // 2)
        tokens = full_tokens[midpoint - n_tokens // 2:midpoint + n_tokens // 2]
        indices = list(range(midpoint - n_tokens // 2, midpoint + n_tokens // 2))
    assert len(tokens) == n_tokens, f"Expected length {n_tokens}, got {len(tokens)}"
    assert len(indices) == n_tokens, f"Expected length {n_tokens}, got {len(indices)}"
    assert tokens == [full_tokens[index] if index != -1 else 0 for index in indices]
    return tokens, indices

class EmptyLabeller():
    def get_label(self, artist=None, genre=None, lyrics=None, total_length=None, offset=None):
        y = np.array([], dtype=np.int64)
        info = dict(artist="n/a", genre="n/a", lyrics=[], full_tokens=[])
        return dict(y=y, info=info)

    def get_batch_labels(self, metas, device='cpu'):
        ys, infos = [], []
        for meta in metas:
            label = self.get_label()
            y, info = label['y'], label['info']
            ys.append(y)
            infos.append(info)

        ys = torch.stack([torch.from_numpy(y) for y in ys], dim=0).to(device).long()
        assert ys.shape[0] == len(metas)
        assert len(infos) == len(metas)
        return dict(y=ys, info=infos)

class Labeller():
    def __init__(self, max_genre_words, n_tokens, sample_length, v3=False):
        self.ag_processor = ArtistGenreProcessor(v3)
        self.text_processor = TextProcessor()
        self.n_tokens = n_tokens
        self.max_genre_words = max_genre_words
        self.sample_length = sample_length
        self.label_shape = (4 + self.max_genre_words + self.n_tokens, )

    def get_label(self, artist, genre, lang, lyrics, total_length, offset):
        artist_id = self.ag_processor.get_artist_id(artist)
        genre_ids = self.ag_processor.get_genre_ids(genre)

        lyrics, _ = self.text_processor.clean(text=f"{lyrics}".strip())
        full_tokens = self.text_processor.tokenise(lyrics)
        tokens, _ = get_relevant_lyric_tokens(full_tokens, self.n_tokens, total_length, offset, self.sample_length)
        
        cptpho_tokens, enroll_x_lens = text_collater([full_tokens])
        cptpho_tokens = cptpho_tokens.squeeze(0)
        lyrics_token_lens = enroll_x_lens[0]
        
        return {
            'artist_name': artist,
            'utt_id': artist_id, # did not use 
            'text': lyrics,
            'audio': None,
            'audio_lens': total_length,
            'audio_features': None,
            'audio_features_lens': None,
            'text_tokens': np.array(cptpho_tokens),
            'text_tokens_lens': lyrics_token_lens,
            'genre': genre,
            'genre_id': genre_ids, # did not use
            'language': language_dict[lang]
        }
        
        '''
        assert len(genre_ids) <= self.max_genre_words
        genre_ids = genre_ids + [-1] * (self.max_genre_words - len(genre_ids))
        y = np.array([total_length, offset, self.sample_length, artist_id, *genre_ids, *tokens], dtype=np.int64)
        assert y.shape == self.label_shape, f"Expected {self.label_shape}, got {y.shape}"
        info = dict(artist=artist, genre=genre, lyrics=lyrics, full_tokens=full_tokens)
        return dict(y=y, info=info)
        '''

    def get_y_from_ids(self, artist_id, genre_ids, lyric_tokens, total_length, offset):
        assert len(genre_ids) <= self.max_genre_words
        genre_ids = genre_ids + [-1] * (self.max_genre_words - len(genre_ids))
        if self.n_tokens > 0:
            assert len(lyric_tokens) == self.n_tokens
        else:
            lyric_tokens = []
        y = np.array([total_length, offset, self.sample_length, artist_id, *genre_ids, *lyric_tokens], dtype=np.int64)
        assert y.shape == self.label_shape, f"Expected {self.label_shape}, got {y.shape}"
        return y