import collections, csv, math, random
import numpy as np

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

        # shuffle both content and labels
        random.seed(seed)
        random.shuffle(content)
        random.seed(seed)
        random.shuffle(labels)

        # encoded is list of list of chars
        self.encoded = [[vocab[k] for k in tweet] for tweet in content]
        self.encoded = self._pad_data(self.encoded)
        self.training_text = self.encoded[0:int(fraction_training * len(self.encoded))]
        self.validation_text = self.encoded[int(fraction_training * len(self.encoded)):]
        self.training_label = labels[0:int(fraction_training * len(self.encoded))]
        self.validation_label = labels[int(fraction_training * len(self.encoded)):]
        self.num_batches = math.ceil(len(self.training_label) / batch_size)
        self.training_batches_text = [self.training_text[i:i+batch_size]
                                      for i in range(0, len(self.training_text), batch_size)]

        self.training_batches_labels = [self.training_label[i:i+batch_size]
                                        for i in range(0, len(self.training_label), batch_size)]

        self.pointer = 0

    def _pad_data(self, data):
        # get max length
        max_length = max([len(l) for l in data])
        self.sequence_length = max_length

        data_padded = [l + [0] * (int(max_length) - len(l)) for l in data]
        return data_padded

    def get_next_training_batch(self):
        x, y = self.training_batches_text[int(self.pointer % self.num_batches)], self.training_batches_labels[int(self.pointer % self.num_batches)]
        self.pointer += 1
        return np.array(x), np.array(y)

    def get_validation_data(self):
        return np.array(self.validation_text), np.array(self.validation_label)


if __name__ == '__main__':
    tweet_data = '/Users/jongan/repositories/twitter-sentiment-analysis/data/oldBrexitSentimentComparison.csv'
    tweetsloader = TweetsDataLoader(tweet_data, 32)
    x, y = tweetsloader.get_next_training_batch()
    valid_x, valid_y = tweetsloader.get_validation_data()
    print (len(x))
    print (len(valid_x))
