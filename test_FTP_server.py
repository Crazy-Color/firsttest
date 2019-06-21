import socket


def main():
    # 1.创建套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 2.地址信息
    server_info = ("", 8080)
    # 3.绑定信息
    server_socket.bind(server_info)
    # 4.开始监听
    server_socket.listen(128)
    # 5.接受客户端命令
    client_socket, client_addr = server_socket.accept()
    file_name = client_socket.recv(1024).decode("utf-8")  # gkb
    back_info =None
    # 6.查找文件信息
    try:
        with open(file_name, "rb") as f:
            back_info = f.read()
    except Exception as result:
        print("文件未找到！")

    if back_info:
        client_socket.send(back_info)
    else:
        print("未找到数据信息！")

    # 7.close
    client_socket.close()

    # 8.close
    server_socket.close()


if __name__ == "__main__":
    main()
