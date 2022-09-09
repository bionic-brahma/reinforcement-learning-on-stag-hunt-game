import socket

SERVER = socket.gethostbyname(socket.gethostname())
PORT = 5051  # The port used by the server

record = "hi"
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SERVER, PORT))
    record = record.encode()
    s.sendall(record)
