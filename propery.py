#####################################
class test(object):
    def __init__(self, price, discount):
        self.org_price = price
        self.discount = discount

    @property
    def prices(self):
        print("price get")
        return self.org_price

    @prices.setter
    def prices(self, value):
        print("change price")
        self.org_price = value
        # return self.oprice

    ###############################
    @prices.deleter
    def prices(self):
        print("del price")
        del self.org_price


obj = test(100, 0.8)
a = obj.prices
obj.prices = 200
del obj.prices
obj.prices

# class Foo:
# #     # def func(self):
# #     #     pass
# #
# #     # 定义property属性
# #     @property
# #     def prop(self, value):
# #         return value
# #
# # # ############### 调用 ###############
# # foo_obj = Foo()
# # # foo_obj.func()  # 调用实例方法
# # foo_obj.prop(200)  # 调用property属性
# # print("")

# help(property)
