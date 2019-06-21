import socket


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    loaclAdr = ("", 8080)

    s.bind(loaclAdr)

    # send_data = input("请输出信息：")

    # ipAdr = ("192.168.2.102", 8080)

    # s.sendto(send_data.encode("utf-8", ipAdr))

    recv_data = s.recvfrom(1024)

    print(recv_data[0].decode("gbk"))

    s.close()


if __name__ == "__main__":
    main()