from queue import Queue, LifoQueue, PriorityQueue
from typing import Optional

from numpy import mean


# 受限队列
class Que(Queue):

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def put(self, item: float, block: bool = ..., timeout: Optional[float] = ...) -> None:
        if self.qsize() >= self.limit:
            super().get()
        super().put(item)

    def mean_value(self):
        return mean(list(self.queue))


if __name__ == '__main__':
    temp = Que(2)
    temp.put(2)
    temp.put(3)
    temp.put(4)
    temp.put(2.4)
    print(temp.mean_value())
