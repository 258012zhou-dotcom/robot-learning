import json
import socket


HOST = "127.0.0.1"
PORT = 50007


def main() -> None:
    sample = {
        "sequence_id": 1,
        "position": [2.0, 1.0],
    }

    with socket.create_connection(
        (HOST, PORT),
        timeout=2.0,
    ) as connection:
        with connection.makefile("rwb") as stream:
            stream.write(
                json.dumps(sample).encode("utf-8") + b"\n"
            )
            stream.flush()

            line = stream.readline()
            if not line:
                raise RuntimeError("server sent no response")

            response = json.loads(line)
            print(f"server response: {response}")


if __name__ == "__main__":
    main()