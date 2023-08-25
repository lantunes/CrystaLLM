import zmq
from lib import CIFScorer


class ZMQScorer(CIFScorer):

    def __init__(self, port: int, timeout_ms: int = 10000):
        self._port = port
        print(f"ZeroMQ CIFScorer using port: {port}")

        # prepare the ZeroMQ context and REQ socket
        context = zmq.Context()
        self._socket = context.socket(zmq.REQ)

        # set the socket timeout
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

        self._socket.connect(f"tcp://localhost:{self._port}")

    def score(self, cif) -> float:
        try:
            # send a request
            self._socket.send_string(cif)

            # wait for the reply
            message = self._socket.recv_string()
            return float(message)

        except zmq.Again as e:
            raise TimeoutError("ZeroMQ request timed out") from e
