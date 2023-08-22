import zmq
from lib import CIFScorer


class ZMQScorer(CIFScorer):

    def __init__(self, port: int):
        self._port = port
        print(f"ZeroMQ CIFScorer using port: {port}")

        # prepare the ZeroMQ context and REQ socket
        context = zmq.Context()
        self._socket = context.socket(zmq.REQ)
        self._socket.connect(f"tcp://localhost:{self._port}")

    def score(self, cif) -> float:
        # send a request and wait for the reply
        self._socket.send_string(cif)
        message = self._socket.recv_string()

        return float(message)
