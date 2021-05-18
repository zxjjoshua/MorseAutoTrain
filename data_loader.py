
class DataLoader:

    def __init__(self, processNodeSet):
        self.processNodeSet = processNodeSet
        self.keys = self.processNodeSet.keys()
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.pos += 1
        # print(self.pos - 1)
        if self.pos >= len(self.processNodeSet):
            self.pos = 1
        return self.processNodeSet[list(self.keys)[self.pos-1]]

    def has_next(self):
        return len(self.keys) > self.pos + 1
