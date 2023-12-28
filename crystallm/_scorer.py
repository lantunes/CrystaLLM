import random

import zmq


class CIFScorer:
    """
    An abstract CIF scorer. A scorer provides a heuristic score for a completed CIF.
    e.g. a scorer might provide the formation energy of the given CIF
    """
    def score(self, cif: str) -> float:
        """
        Returns a score for the CIF. A higher score is better.

        :param cif: the CIF to be scored
        :returns: a float representing the score
        """
        pass


class RandomScorer(CIFScorer):

    def __init__(self, min_score: float = -5., max_score: float = 5., seed: int = None):
        """
        A RandomScorer returns a random score. This scorer is intended as a substitute for a
        true scorer for demonstration purposes, in scenarios where a true scorer is not available.

        :param min_score: the minimum score to be returned
        :param max_score: the maximum score to be returned
        :param seed: a random seed (optional)
        """
        self._local_random = random.Random(seed)
        self._min_score = min_score
        self._max_score = max_score

    def score(self, cif: str) -> float:
        return self._local_random.uniform(self._min_score, self._max_score)


class ZMQScorer(CIFScorer):

    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 10000):
        """
        A CIF scorer which returns a score obtained from another process via ZMQ.

        :param host: the ZMQ host
        :param port: the ZMQ port
        :param timeout_ms: the ZMQ socket timeout in milliseconds
        """
        print(f"ZeroMQ CIFScorer using host:port: {host}:{port}")

        # prepare the ZeroMQ context and REQ socket
        context = zmq.Context()
        self._socket = context.socket(zmq.REQ)

        # set the socket timeout
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

        self._socket.connect(f"tcp://{host}:{port}")

    def score(self, cif: str) -> float:
        try:
            # send a request
            self._socket.send_string(cif)

            # wait for the reply
            message = self._socket.recv_string()
            return float(message)

        except zmq.Again as e:
            raise TimeoutError("ZeroMQ request timed out") from e
