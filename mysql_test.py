from pymysql import *


# conn = connect(host='localhost',port=3306,database='jing_dong',user='root',password='hmn123',charset='utf8')
# cou = conn.cursor()
# info = cou.execute("select * from goods;")
# infos = cou.fetchmany(3)
# for item in infos:
#     print(item)


class mysql:
    def __init__(self):
        self.conn = connect(host='localhost',port=3306,database='jing_dong',user='root',password='hmn123',charset='utf8')
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    def show_all_info(self,sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def add_info(self, sql):
        self.cursor.execute(sql)
        self.conn.commit()

    @staticmethod
    def menu():
        print("-"*20)
        print("1：商品")
        print("2.目录")
        print("3.品牌")
        print("4：退出")
        print("5:添加商品信息")
        print("-"*20)
        return input("请选择:")


def main():
    test_user = mysql()

    while True:
        num = test_user.menu()
        if num == "1":
            sql = "select * from goods"
            print(test_user.show_all_info(sql))
        elif num == "2":
            sql = "select name from goods_cates"
            print(test_user.show_all_info(sql))
        elif num == "3":
            sql = "select name from goods_brands"
            print(test_user.show_all_info(sql))
        elif num == "4":
            break
        elif num == "5":
            sql = "insert into goods_cates (name) value(\"大宝\")"
            test_user.add_info(sql)
        else:
            print("请重新输入！")
            continue

        # for item in test_user.show_all_info(sql):
        #     print(item)


if __name__ =="__main__":
    main()