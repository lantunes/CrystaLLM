import zmq


if __name__ == '__main__':
    zmq_port = 5555
    cif_fname = "../out/manual_tests_cif_model_33/LiTa2NiSe5/LiTa2NiSe5_generated_large.cif"

    print(f"using port: {zmq_port}")

    print(f"reading CIF: {cif_fname}")
    with open(cif_fname, "rt") as f:
        cif = f.read()

    # Prepare the ZeroMQ context and REQ socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{zmq_port}")

    # Send a request and wait for the reply
    socket.send_string(cif)
    message = socket.recv_string()
    print(f"received reply: {float(message):.3f}")
