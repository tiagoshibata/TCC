#!/usr/bin/env python3
import os
import queue
import threading


def consumer_thread(queue, function):
    while True:
        job = queue.get()
        if not job:
            break
        function(job)
        queue.task_done()


class ConsumerPool:
    def __init__(self, function, num_workers=0, queue_size=2):
        if not num_workers:
            num_workers = len(os.sched_getaffinity(0))
        self.queue = queue.Queue(queue_size)
        self.pool = [threading.Thread(target=lambda: consumer_thread(self.queue, function)) for _ in range(num_workers)]
        for thread in self.pool:
            thread.start()

        # ConsumerPool.put will put into its queue
        self.put = self.queue.put

    def join(self):
        self.queue.join()
        for _ in self.pool:
            self.queue.put(None)
        for thread in self.pool:
            thread.join()
