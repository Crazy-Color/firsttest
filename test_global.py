# import threading
# import time
#
# g_num = 100
#
#
# def test_print():
#     global g_num
#     print("打印数据：%d" % g_num)
#
#
# def test_change():
#     global g_num
#     for i in range(5):
#         g_num += 1
#         print("测试数据：%d" % g_num)
#
#
# if __name__ == "__main__":
#     t1 = threading.Thread(target=test_print)
#     t2 = threading.Thread(target=test_change)
#     t1.start()
#     time.sleep(2)
#     # t2 = threading.Thread(target=test_change())
#     t2.start()
#
#     print(threading.enumerate())


from threading import Thread
import time


def work1(nums):
    nums.append(44)
    print("----in work1---",nums)


def work2(nums):
    # 延时一会，保证t1线程中的事情做完
    # time.sleep(1)
    print("----in work2---",nums)


g_nums = [11, 22, 33]

t1 = Thread(target=work1, args=(g_nums,))
t2 = Thread(target=work2, args=(g_nums,))
t2.start()

time.sleep(1)
t1.start()

# import threading
# import time
#
# g_num2 = 0
# mutex = threading.Lock()
#
#
# def add1(num):
#     global g_num2
#     mutex.acquire()
#     for i in range(num):
#         g_num2 += 1
#     mutex.release()
#     print("add1:%d" % g_num2)
#
#
# def add2(num):
#     global g_num2
#     mutex.acquire()
#     for i in range(num):
#         g_num2 += 1
#     mutex.release()
#     print("add2:%d" % g_num2)
#
#
# def main():
#     t1 = threading.Thread(target=add1, args=(1000000,))
#     t2 = threading.Thread(target=add2, args=(1000000,))
#
#     t1.start()
#     t2.start()
#
#     time.sleep(2)
#     print("主程序结束：%d" % g_num2)


# if __name__ == "__main__":
#     main()
