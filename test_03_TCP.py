import socket


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # ip = "192.168.1.2"
    # port = 8080
    localaddr = ("", 8080)
    # s.connect(localaddr)
    s.bind(localaddr)
    s.listen(128)

    while True:
        n_socket,a_addr = s.accept()
        # data = input("请输入数据：")
        # s.send(data.encode("utf-8"))
        while True:
            data = n_socket.recv(1024)
            if data:
                print(data.decode("gbk"))
                n_socket.send("接收到".encode("utf-8"))
            else:
                break

        n_socket.close()


    s.close()


if __name__ == "__main__":
    main()