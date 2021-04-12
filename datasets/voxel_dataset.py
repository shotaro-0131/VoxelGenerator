class DataSet:
    def __init__(self, train, label):
        self.train = train
        self.label = label

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        return self.train[index], self.label[index]
