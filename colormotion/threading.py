#!/usr/bin/env python3
import os
import queue
from threading import Thread


default_worker_count = len(os.sched_getaffinity(0))


class ThreadPool:
    def __init__(self, function, num_workers=default_worker_count, queue_size=2):
        self.function = function
        self.queue = queue.Queue(queue_size)
        self.pool = [Thread(target=self.thread) for _ in range(num_workers)]
        for thread in self.pool:
            thread.start()
        self.running = True

    def thread(self):
        raise NotImplementedError()

    def join(self):
        for thread in self.pool:
            thread.join()


class ConsumerPool(ThreadPool):
    def __init__(self, function, num_workers=default_worker_count, queue_size=2):
        super().__init__(function, num_workers, queue_size)

        # ConsumerPool.put will put into its queue
        self.put = self.queue.put

    def thread(self):
        while True:
            job = self.queue.get()
            if not job:
                break
            self.function(job)
            self.queue.task_done()

    def join(self):
        self.queue.join()
        for _ in self.pool:
            self.queue.put(None)
        super().join()


class ProducerPool(ThreadPool):
    '''Pool of threads producing items into a queue.'''
    def __init__(self, function, num_workers=default_worker_count, queue_size=2):
        self.running = True
        # Have at least num_workers slots to fill when joining with threads
        # (in case all threads are still generating an item)
        queue_size = max(queue_size, num_workers)
        super().__init__(function, num_workers, queue_size)

        # ProducerPool.get and ProducerPool.task_done will call
        # corresponding functions from its queue
        self.get = self.queue.get
        self.task_done = self.queue.task_done

    def __iter__(self):
        return self

    def __next__(self):
        item = self.get()
        if item is None:
            raise StopIteration()
        return item

    def thread(self):
        generator = self.function()
        try:
            while self.running:
                self.queue.put(next(generator))
        except StopIteration:
            pass

    def clear_queue(self):
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass

    def join(self):
        '''Join with all threads.

        Might leave leftover items in the queue.
        '''
        self.running = False
        self.clear_queue()
        super().join()
