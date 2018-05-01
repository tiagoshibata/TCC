from colormotion.threading import ProducerPool

def test_ProducerPool():
    def producer():
        for i in range(3):
            yield i

    producer_pool = ProducerPool(producer, num_workers=1)
    for i, produced in zip(producer_pool, range(3)):
        assert i == produced
    producer_pool.join()
