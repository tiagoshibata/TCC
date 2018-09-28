from contextlib import contextmanager

import tensorflow as tf


@contextmanager
def new_model_session():
    graph = tf.Graph()
    with graph.as_default():
        session = tf.Session()
        with session.as_default():
            yield model_session((graph, session))


@contextmanager
def model_session(graph_and_session):
    graph, session = graph_and_session
    with graph.as_default():
        with session.as_default():
            yield
