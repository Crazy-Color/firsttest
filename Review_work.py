# 测试程序及改进方案

##################################
# 1.hello world
# .idea目录下面保存着项目相关的信息（解释版本、项目文件等）
# linux下面执行./pycharm.sh
# python的编译器Eclipse + PyDev、Sublime Text、Visual Studio Code、Spyder（专业数据分析）

# 2.注释
# 单行：#
# 多行：""""""

# 3.算数运算符
# def calculation():
#     num_a = 6
#     num_b = 20
#     num_sum = num_a + num_b
#     num_dif = num_b - num_a
#     num_times = num_b * num_a
#     num_odd = num_b /num_a
#     num_block = num_b // num_a
#     num_last = num_b % num_a

# 4.变量
# type(name)：查看变量名数据类型
# input("提示输入：")

# demo_list = [1, 2, 3]
# print("定义列表后的内存地址 %d" % id(demo_list))
# demo_list.append(999)
# demo_list.pop(0)
# demo_list.remove(2)
# demo_list[0] = 10
# print("修改数据后的内存地址 %d" % id(demo_list))
# demo_dict = {"name": "小明"}
# print("定义字典后的内存地址 %d" % id(demo_dict))
# demo_dict["age"] = 18
# demo_dict.pop("name")
# demo_dict["name"] = "老王"
# print("修改数据后的内存地址 %d" % id(demo_dict))

# -*- coding:utf-8 -*-
# from multiprocessing import Process
# import os
# from time import sleep
#
#
# def run_proc(name, age, **kwargs):
#     for i in range(10):
#         print('子进程运行中，name= %s,age=%d ,pid=%d...' % (name, age, os.getpid()))
#         print(kwargs)
#         sleep(0.2)
#
#
# if __name__=='__main__':
#     p = Process(target=run_proc, args=('test', 18, {"m":20}))
#     p.start()
#     sleep(1)  # 1秒中之后，立即结束子进程
#     p.terminate()
#     p.join()


# class FibIterator(object):
#     """斐波那契数列迭代器"""
#     def __init__(self, n):
#         """
#         :param n: int, 指明生成数列的前n个数
#         """
#         self.n = n
#         # current用来保存当前生成到数列中的第几个数了
#         self.current = 0
#         # num1用来保存前前一个数，初始值为数列中的第一个数0
#         self.num1 = 0
#         # num2用来保存前一个数，初始值为数列中的第二个数1
#         self.num2 = 1
#
#     def __next__(self):
#         """被next()函数调用来获取下一个数"""
#         if self.current < self.n:
#             num = self.num1
#             self.num1, self.num2 = self.num2, self.num1+self.num2
#             self.current += 1
#             return num
#         else:
#             raise StopIteration
#
#     def __iter__(self):
#         """迭代器的__iter__返回自身即可"""
#         return self
#
#
# if __name__ == '__main__':
#     fib = FibIterator(10)
#     for num in fib:
#         print(num, end=" ")
import copy

a = [11,22,33]
b = copy.copy(a)
print(id(a))

print(id(b))

a.append(44)
print(a)
print(id(a))
print(b)
print(b)