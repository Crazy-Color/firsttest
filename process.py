import multiprocessing
import time


def creat_data(q):
    data = [1, 2, 3, 4, 5]
    for temp in data:
        q.put(temp)
        # print(temp)

    print("装载完成")


def del_data(q):
    data = list()
    while True:
        data.append(q.get())
        if q.empty():
            break
    print(data)


def main():
    q = multiprocessing.Queue(5)
    c_data = multiprocessing.Process(target=creat_data, args=(q,))
    d_data = multiprocessing.Process(target=del_data, args=(q,))

    c_data.start()
    time.sleep(1)
    d_data.start()


if __name__ == "__main__":
    main()