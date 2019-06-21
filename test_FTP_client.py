import socket


def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip = input("请输出服务器IP：")
    port = int(input("请输入服务器端口："))
    server_info = (ip, port)
    # 1.连接服务器
    client_socket.connect(server_info)
    # 2.发送请求，下载文件
    file_name = input("请输入需要下载的文件：")
    client_socket.send(file_name.encode("utf-8"))
    # 3.下载文件处理
    recv_data = client_socket.recv(1024)
    if recv_data:
        with open("[new]"+file_name, "wb") as f:
            f.write(recv_data)
    else:
        print("下载失败！")


if __name__ == "__main__":
    main()