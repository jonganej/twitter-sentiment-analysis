import collections, csv, math, random
import numpy as np
import tensorflow as tf

class TweetsDataLoader:
    def __init__(self, filename_csv, batch_size, fraction_training=0.8, seed=42):
        content, labels = [], []
        with open(filename_csv, 'r') as f:
            csvreader = csv.DictReader(f, delimiter=',')
            for row in csvreader:
                if row['gs_sentiment'] == 'stay' or row['gs_sentiment'] == 'leave':
                    label_class = int(row['gs_sentiment'] == 'stay')
                    labels.append([int(i==label_class) for i in range(2)])
                    content.append(row['text'])
        text = " ".join(content)
        counter = collections.Counter(text)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        chars, _ = zip(*count_pairs)
        vocab = dict(zip(chars, range(len(chars))))
        self.vocab = vocab
        # shuffle both content and labels
        random.seed(seed)
        random.shuffle(content)
        random.seed(seed)
        random.shuffle(labels)

        # encoded is list of list of chars
        self.encoded = [[vocab[k] for k in tweet] for tweet in content]
        self.original_seq_length = [len(t) for t in self.encoded]
        self.encoded = self._pad_data(self.encoded)
        self.training_text, self.validation_text = self._divide_data(self.encoded, fraction_training)
        self.training_label, self.validation_label = self._divide_data(labels, fraction_training)
        self.training_seq_lengths, self.validation_seq_lengths = self._divide_data(self.original_seq_length, fraction_training)
        self.num_batches = math.ceil(len(self.training_label) / batch_size)
        self.training_batches_text = self._create_batches(self.training_text, batch_size)

        self.training_batches_labels = self._create_batches(self.training_label, batch_size)

        self.training_batches_seqlength = self._create_batches(self.training_seq_lengths, batch_size)
        self.pointer = 0

    def _divide_data(self, data, frac_training):
        divide_idx = int(frac_training * len(data))
        return data[0:divide_idx], data[divide_idx:]

    def _create_batches(self, data, batch_size):
        return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    def _pad_data(self, data):
        # get max length
        max_length = max([len(l) for l in data])
        self.sequence_length = max_length

        data_padded = [l + [0] * (int(max_length) - len(l)) for l in data]
        return data_padded

    def get_next_training_batch(self):
        x, y = self.training_batches_text[int(self.pointer % self.num_batches)], self.training_batches_labels[int(self.pointer % self.num_batches)]
        seq_lengths = self.training_batches_seqlength[int(self.pointer % self.num_batches)]
        self.pointer += 1
        return np.array(x), np.array(y), np.array(seq_lengths)

    def get_validation_data(self):
        return np.array(self.validation_text), np.array(self.validation_label)

    def reset_pointer(self):
        self.pointer = 0

if __name__ == '__main__':
    tweet_data = '/home/ganenjij/repositories/twitter-sentiment-analysis/data/oldBrexitSentimentComparison.csv'
    tweetsloader = TweetsDataLoader(tweet_data, 32)
    x, y, seq_length = tweetsloader.get_next_training_batch()
    valid_x, valid_y = tweetsloader.get_validation_data()
    print (len(x))
    print (len(valid_x))
