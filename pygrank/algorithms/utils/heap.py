import heapq as heapq


class Heap:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self._counter = 0

    def __contains__(self, item):
        return item in self.entry_finder

    def __iter__(self):
        return self

    def __next__(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not None:
                del self.entry_finder[item]
                return item
        raise StopIteration()

    def __len__(self):
        return len(self.pq)

    def remove(self, item):
        entry = self.entry_finder[item]
        entry[-1] = None

    def add(self, item, priority=0):
        if item in self:
            self.remove(item)
        entry = [priority, self._counter, item]
        heapq.heappush(self.pq, entry)
        self.entry_finder[item] = entry
        self._counter += 1