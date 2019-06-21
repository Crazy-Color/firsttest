import socket


def sent_msg(udp_socket):
    ip = input("请输入地址：")
    port = input("请输入端口：")
    send_data = input("请输入传输数据：")
    udp_socket.sendto(send_data.encode("utf-8"), (ip, int(port)))


def recv_msg(udp_socket):
    recv_data = udp_socket.recvfrom(1024)
    print(recv_data[0].decode("gbk"))


def main():

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind("", 7890)

    while True:
        order = input("请输入数据类型（0-send,1-recv,2-exit）：")

        if order == 0:
            sent_msg(s)
        elif order == 1:
            recv_msg(s)
        elif order == 2:
            break
        else:
            print("请重新输入命令！")

    s.close()
    print("abc")
    print("abcd")
