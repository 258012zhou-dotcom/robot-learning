import json
import socket


HOST = "127.0.0.1"
PORT = 50007


def main() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        print(f"server listening on {HOST}:{PORT}")

        connection, address = server.accept()

        with connection:
            print(f"client connected: {address}")

            with connection.makefile("rwb") as stream:
                line = stream.readline()
                if not line:
                    raise RuntimeError("client sent no data")

                sample = json.loads(line)
                print(f"received sample: {sample}")

                response = {
                    "status": "ok",
                    "sequence_id": sample["sequence_id"],
                }

                stream.write(
                    json.dumps(response).encode("utf-8") + b"\n"
                )
                stream.flush()


if __name__ == "__main__":
    main()