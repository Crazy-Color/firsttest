# from collections import Iterable
# from collections import Iterable


class Fib(object):
    def __init__(self, n):
        self.n = n
        self.current = 0
        self.num1 = 1
        self.num2 = 1

    def __next__(self):
        if self.current < self.n:
            temp2 = self.num1
            self.num1, self.num2 = self.num2, self.num1 + self.num2
            self.current += 1
            return temp2
        else:
            raise StopIteration

    def __iter__(self):
        return self


if __name__ == "__main__":
    fib = Fib(10)
    # print(isinstance(fib, Iterable))
    for temp in fib:
        print(temp)