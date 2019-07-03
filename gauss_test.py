# print("abc")
# from math import *
# #
# #
# # def Gauss(mu,sigma2,x):
# #     return 1/sqrt(2.*pi*sigma2)*exp(-.5*(x-mu)**2/sigma2)
# #
# #
# # if __name__ == "__main__":
# #     res = Gauss(10.,4.,8.)
# #     print("高斯公式计算结果：%.06f" %res)

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
#
# # Read in the image
# image = mpimg.imread('test.png')
#
# # Grab the x and y size and make a copy of the image
# ysize = image.shape[0]
# xsize = image.shape[1]
# color_select = np.copy(image)
#
# # Define color selection criteria
# ###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
# red_threshold = 0
# green_threshold = 0
# blue_threshold = 0
# ######
#
# rgb_threshold = [red_threshold, green_threshold, blue_threshold]
#
# # Do a boolean or with the "|" character to identify
# # pixels below the thresholds
# thresholds = (image[:,:,0] < rgb_threshold[0]) \
#             | (image[:,:,1] < rgb_threshold[1]) \
#             | (image[:,:,2] < rgb_threshold[2])
# color_select[thresholds] = [0,0,0]
#
# # Display the image
# plt.imshow(color_select)
# plt.show()
#
# # Uncomment the following code if you are running the code locally and wish to save the image
# # mpimg.imsave("test-after.png", color_select)

# from math import *


# def Guss2(mean1,var1,mean2,var2):
#     new_mean = (mean1*var2 + mean2*var1)/(var1 + var2)
#     new_var = (var1*var2)/(var1+var2)
#     return [new_mean,new_var]
#
#
# if __name__ == "__main__":
#     print(Guss2(10.,8.,13.,2.))


# from matplotlib import pyplot as plt
# import random
# from matplotlib import font_manager


# -*- coding:utf-8 -*-

# x=range(2,26,2)
# y=[15,13,14.5,17,20,25,26,26,24,22,18,15]
#
# plt.figure(figsize=(20,8),dpi=80)
# plt.plot(x,y)
# plt.savefig("./t1.png")
# plt.show()
# my_font = font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=15)
# x=range(0,120)
# y=[random.randint(20,35) for i in range(0,120)]
# plt.figure(figsize=(20,8),dpi=80)
# plt.plot(x,y)
#
# x_ticks = ["10点{}分".format(i) for i in range(60)]
# x_ticks += ["11点{}分".format(i) for i in range(60)]
# plt.xticks(list(x)[::5], x_ticks[::5], rotation=90,fontproperties=my_font)
#
# plt.xlabel("时间",fontproperties=my_font)
# plt.ylabel("温度（℃）",fontproperties=my_font)
# plt.title("变化情况",fontproperties=my_font)
# plt.grid(alpha=0.4)
# plt.show()

# # -*- coding=utf-8 -*-
# # Kalman filter example demo in Python
#
# # A Python implementation of the example given in pages 11-15 of "An
# # Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# # University of North Carolina at Chapel Hill, Department of Computer
# # Science, TR 95-041,
# # http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
#
# # by Andrew D. Straw
# #coding:utf-8
# import numpy
# import pylab
# import math
#
# #这里是假设A=1，H=1的情况
#
# # intial parameters
# n_iter = 50
# sz = (n_iter,) # size of array
# x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
# z = numpy.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)
#
# Q = 1e-5 # process variance
#
# # allocate space for arrays
# xhat=numpy.zeros(sz)      # a posteri estimate of x
# P=numpy.zeros(sz)         # a posteri error estimate
# xhatminus=numpy.zeros(sz) # a priori estimate of x
# Pminus=numpy.zeros(sz)    # a priori error estimate
# K=numpy.zeros(sz)         # gain or blending factor
#
# R = 0.1**2 # estimate of measurement variance, change to see effect
#
# # intial guesses
# xhat[0] = 0.0
# P[0] = 1.0
#
# for k in range(1,n_iter):
#     # time update
#     xhatminus[k] = xhat[k-1]  #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
#     Pminus[k] = P[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
#
#     # measurement update
#     K[k] = Pminus[k]/( Pminus[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
#     xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
#     P[k] = (1-K[k])*Pminus[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
#
# pylab.figure()
# pylab.plot(z,'k+',label='noisy measurements')     #测量值
# pylab.plot(xhat,'b-',label='a posteri estimate')  #过滤后的值
# pylab.axhline(x,color='g',label='truth value')    #系统值
# pylab.legend()
# pylab.xlabel('Iteration')
# pylab.ylabel('Voltage')
#
# pylab.figure()
# valid_iter = range(1,n_iter) # Pminus not valid at step 0
# pylab.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
# pylab.xlabel('Iteration')
# pylab.ylabel('$(Voltage)^2$')
# pylab.setp(pylab.gca(),'ylim',[0,.01])
# pylab.show()

# a=math.degrees(math.pi / 3)
# print(a)

# # !/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Dec 18 19:37:13 2018
# @author: sc
# args explanition:
#     p:covariance
#     x:state
#     z:observation
#     u:control
#     Pred:predict
#     Est:estatemation
#     Q：表示过程激励噪声的协方差，它是状态转移矩阵与实际过程之间的误差。
#     R：表示测量噪声协方差，和仪器相关的一个特性，
# """
# import numpy as np
# import math
# import matplotlib.pyplot as plt
#
# Q = np.diag([0.1, 0.1, math.radians(1.0), 1.0]) ** 2
# R = np.diag([1.0, math.radians(40.0)]) ** 2
# dt = 0.1
#
#
# def motion_model(x, u):
#     B = np.matrix([[dt * math.cos(x[2, 0]), 0.0],
#                    [dt * math.sin(x[2, 0]), 0.0],
#                    [0.0, dt],
#                    [1.0, 0.0]])
#     x = x + B * u
#     return x
#
#
# # def observe_model(z):
# #    H=
# #    pass
# # def JacoMo(xEst,u):
# #    return jMo
# # def JacoOb(xEst):
# #    return jOb
# def JacoMo(x, u):
#     """
#     Jacobian of Motion Model
#     motion model
#     x_{t+1} = x_t+v*dt*cos(yaw)
#     y_{t+1} = y_t+v*dt*sin(yaw)
#     yaw_{t+1} = yaw_t+omega*dt
#     v_{t+1} = v{t}
#     so
#     dx/dyaw = -v*dt*sin(yaw)
#     dx/dv = dt*cos(yaw)
#     dy/dyaw = v*dt*cos(yaw)
#     dy/dv = dt*sin(yaw)
#     """
#     yaw = x[2, 0]
#     v = u[0, 0]
#     jF = np.matrix([[1.0, 0.0, -dt * v * math.sin(yaw), dt * math.cos(yaw)],
#                     [0.0, 1.0, dt * v * math.cos(yaw), dt * math.sin(yaw)],
#                     [0.0, 0.0, 1.0, 0.0],
#                     [0.0, 0.0, 0.0, 1.0]])
#
#     return jF
#
#
# def JacoOb(x):
#     # Jacobian of Observation Model
#     jH = np.matrix([
#         [1, 0, 0, 0],
#         [0, 1, 0, 0]
#     ])
#
#     return jH
#
#
# def ekf_estimation(xEst, pEst, z, u):
#     # yu ce predict
#     xPre = motion_model(xEst, u)
#
#     jMo = JacoMo(xEst, u)  # Jacobin of motion model
#     jOb = JacoOb(xEst)
#     pPre = jMo * pEst * jMo.T + Q
#     s = jOb * pPre * jOb.T + R
#     k = pPre * jOb.T * np.linalg.inv(s)
#     # update estimate
#     H = np.matrix([
#         [1, 0, 0, 0],
#         [0, 1, 0, 0]
#     ])
#     zPre = H * xPre
#     xEst = xPre + k * (z - zPre)
#     pEst = (np.eye(len(xEst)) - k * jOb) * pPre
#     return xEst, pEst
#
#
# def input_data(xTrue, xImu):
#     u = np.matrix([1.0, 0.1]).T  # jiaosudu he xiansudu
#     xTrue = motion_model(xTrue, u)
#     Qsim = np.diag([0.5, 0.5]) ** 2
#     Rsim = np.diag([1.0, math.radians(5.0)]) ** 2
#
#     # GPS
#     z = np.matrix([xTrue[0, 0] + np.random.randn() * Qsim[0, 0],
#                    xTrue[1, 0] + np.random.randn() * Qsim[1, 1]]).T
#     ud = np.matrix([u[0, 0] + np.random.randn() * Rsim[0, 0],
#                     u[1, 0] + np.random.randn() * Rsim[1, 1]]).T
#     xImu = motion_model(xImu, ud)
#     return ud, z, xTrue, xImu
#
#
# if __name__ == "__main__":
#     xEst = np.matrix(np.zeros((4, 1)))
#     xTrue = xEst
#     xImu = xTrue
#     pEst = np.eye(4)
#     t = 0.1
#
#     hTrue = xTrue
#     hEst = xEst
#     hz = np.zeros((2, 1))
#     hImu = xImu
#     for i in range(1000):
#         u, z, xTrue, xImu = input_data(xTrue, xImu)
#
#         xEst, pEst = ekf_estimation(xEst, pEst, z, u)
#         hTrue = np.hstack((hTrue, xTrue))
#         hEst = np.hstack((hEst, xEst))
#         hImu = np.hstack((hImu, xImu))
#         hz = np.hstack((hz, z))
#         # plot
#         plt.cla()
#         plt.plot(hz[0, :], hz[1, :], ".g")
#         plt.plot(np.array(hTrue[0, :]).flatten(),
#                  np.array(hTrue[1, :]).flatten(), "-b")
#         plt.plot(np.array(hEst[0, :]).flatten(),
#                  np.array(hEst[1, :]).flatten(), "-r")
#         plt.plot(np.array(hImu[0, :]).flatten(),
#                  np.array(hImu[1, :]).flatten(), "-k")
#         plt.pause(0.001)

# from matplotlib import pyplot as plt
# from matplotlib import font_manager


# my_font = font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=15)
# 设置散点图
# y_3 = [11,17,16,11,12,11,12,6,6,7,8,9,12,15,14,17,18,21,16,17,20,14,15,15,15,19,21,22,22,22,23]
# y_10 = [26,26,28,19,21,17,16,19,18,20,20,19,22,23,17,20,21,20,22,15,11,15,5,13,17,10,11,13,12,13,6]
#
# x_3 = list(range(1,32))
# x_10 = list(range(51,82))
# _xticks = x_3 + x_10
# _xticksname = ["3月{}日".format(i) for i in x_3]
# _xticksname += ["10月{}日".format(i-50) for i in x_10]
# plt.figure(figsize=(20,8),dpi=80)
# plt.scatter(x_3,y_3,label="x")
# plt.scatter(x_10,y_10,label="y")
# plt.xticks(_xticks[::3],_xticksname[::3],rotation=90,fontproperties=my_font)
# plt.xlabel("时间",fontproperties=my_font)
# plt.ylabel("气温",fontproperties=my_font)
# plt.title("气温变化图",fontproperties=my_font)
# plt.legend(prop=my_font,loc="upper left")
# plt.show()

# 设置条形图
# a = ["战狼2","速度与激情8","功夫瑜伽","西游伏妖篇","变形金刚5：最后的骑士","摔跤吧！爸爸","加勒比海盗5：死无对证","金刚：骷髅岛","极限特工：终极回归","生化危机6：终章","乘风破浪","神偷奶爸3","智取威虎山","大闹天竺","金刚狼3：殊死一战","蜘蛛侠：英雄归来","悟空传","银河护卫队2","情圣","新木乃伊",]
# b = [56.01,26.94,17.53,16.49,15.45,12.96,11.8,11.61,11.28,11.12,10.49,10.3,8.75,7.55,7.32,6.99,6.88,6.86,6.58,6.23]
# plt.figure(figsize=(20,8),dpi=80)
# plt.grid(alpha=0.3)
# plt.bar(range(len(a)),b,width=0.3)
# plt.xticks(range(len(a)),a,rotation=90,fontproperties=my_font)
# plt.show()

# a = ["战狼2","速度与激情8","功夫瑜伽","西游伏妖篇","变形金刚5：最后的骑士","摔跤吧！爸爸","加勒比海盗5：死无对证","金刚：骷髅岛","极限特工：终极回归","生化危机6：终章","乘风破浪","神偷奶爸3","智取威虎山","大闹天竺","金刚狼3：殊死一战","蜘蛛侠：英雄归来","悟空传","银河护卫队2","情圣","新木乃伊",]
# b = [56.01,26.94,17.53,16.49,15.45,12.96,11.8,11.61,11.28,11.12,10.49,10.3,8.75,7.55,7.32,6.99,6.88,6.86,6.58,6.23]
# plt.figure(figsize=(20,8),dpi=80)
# plt.grid(alpha=0.5)
# plt.barh(range(len(a)),b,height=0.3,color="orange")
# plt.yticks(range(len(a)),a,fontproperties=my_font)
# plt.show()

# a = ["猩球崛起3：终极之战","敦刻尔克","蜘蛛侠：英雄归来","战狼2"]
# b_16 = [15746,312,4497,319]
# b_15 = [12357,156,2045,168]
# b_14 = [2358,399,2358,362]
#
# bar_width = 0.2
# x_14 = list(range(len(a)))
# x_15 = [i+bar_width for i in x_14]
# x_16 = [i+bar_width for i in x_15]
#
# plt.figure(figsize=(20,8),dpi=80)
# plt.grid(alpha=.5)
# plt.bar(x_14,b_14,bar_width,label="14号")
# plt.bar(x_15,b_15,bar_width,label="15号")
# plt.bar(x_16,b_16,bar_width,label="16号")
# plt.legend(prop=my_font,loc=0)
# plt.xticks(x_15,a,fontproperties=my_font)
# # plt.bar(range(len(a)),b_15,width=0.1,color="orange")
# plt.show()

# 绘制直方图
# a=[131,  98, 125, 131, 124, 139, 131, 117, 128, 108, 135, 138, 131, 102, 107, 114, 119, 128, 121, 142, 127, 130, 124, 101, 110, 116, 117, 110, 128, 128, 115,  99, 136, 126, 134,  95, 138, 117, 111,78, 132, 124, 113, 150, 110, 117,  86,  95, 144, 105, 126, 130,126, 130, 126, 116, 123, 106, 112, 138, 123,  86, 101,  99, 136,123, 117, 119, 105, 137, 123, 128, 125, 104, 109, 134, 125, 127,105, 120, 107, 129, 116, 108, 132, 103, 136, 118, 102, 120, 114,105, 115, 132, 145, 119, 121, 112, 139, 125, 138, 109, 132, 134,156, 106, 117, 127, 144, 139, 139, 119, 140,  83, 110, 102,123,107, 143, 115, 136, 118, 139, 123, 112, 118, 125, 109, 119, 133,112, 114, 122, 109, 106, 123, 116, 131, 127, 115, 118, 112, 135,115, 146, 137, 116, 103, 144,  83, 123, 111, 110, 111, 100, 154,136, 100, 118, 119, 133, 134, 106, 129, 126, 110, 111, 109, 141,120, 117, 106, 149, 122, 122, 110, 118, 127, 121, 114, 125, 126,114, 140, 103, 130, 141, 117, 106, 114, 121, 114, 133, 137,  92,121, 112, 146,  97, 137, 105,  98, 117, 112,  81,  97, 139, 113,134, 106, 144, 110, 137, 137, 111, 104, 117, 100, 111, 101, 110,105, 129, 137, 112, 120, 113, 133, 112,  83,  94, 146, 133, 101,131, 116, 111,  84, 137, 115, 122, 106, 144, 109, 123, 116, 111,111, 133, 150]
#
# d= 3
# num_bins = (max(a)-min(a))//d
#
# plt.figure(figsize=(20,8),dpi=80)
# plt.xticks(range(min(a),max(a)+d,d))
# plt.xlabel("分布范围",fontproperties=my_font)
# plt.ylabel("分布频率",fontproperties=my_font)
# plt.title("数据分布情况",fontproperties=my_font)
# plt.grid(alpha=0.5)
#
# plt.hist(a,num_bins,color="black",normed=True)
# plt.show()

# 绘制直方图2（bar）
# interval = [0,5,10,15,20,25,30,35,40,45,60,90]
# width = [5,5,5,5,5,5,5,5,5,15,30,60]
# quantity = [836,2737,3723,3926,3596,1438,3273,642,824,613,215,47]
#
# plt.figure(figsize=(20,8),dpi=80)
# plt.xticks([i-0.5 for i in range(13)],interval+[interval[-1]+width[-1]])
# plt.bar(range(12),quantity,width=1,color="black")
# plt.grid(True,alpha=0.5)
# plt.show()

# from pyecharts import Funnel
# attr =["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
# value =[20, 40, 60, 80, 100, 120]
# funnel =Funnel("漏斗图示例")
# funnel.add("商品", attr, value, is_label_show=True, label_pos="inside", label_text_color="#fff")
# funnel.render()

# from pyecharts.charts import Bar
# from pyecharts.charts import Funnel
#
# # from pyecharts import Bar
# barx = Bar()
# barx.add()
# barx.show_config()
# barx.render()

# import seaborn as sns
# import plotly.plotly as py

# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import font_manager
# a = np.arange(1,7)
# print(a)
# print(type(a),a.dtype)
# b = a.astype(np.float)
# print(b,type(b),b.dtype)
# c = a.reshape(2,3)
# print(c,type(c),c.dtype)
# d = c.flatten()
# print(d,type(d),d.dtype)
# e = d.astype(np.int8)
# print(e,type(e),e.dtype)

# t1 = np.loadtxt("GB_video_data_numbers.csv",delimiter=",",dtype=int,unpack=False)
# t2 = np.loadtxt("US_video_data_numbers.csv",delimiter=",",dtype=int,unpack=False)
# my_font = font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=15)

# t1 = t1[t1[:,-1]<= 5000][:,-1]
# d = 500
# num_bins = (t1.max()-t1.min())//d
# plt.figure(figsize=(20,8),dpi=80)
# # plt.xticks()
# plt.xlabel("x轴",fontproperties=my_font)
# plt.ylabel("y轴",fontproperties=my_font)
# plt.title("数据显示",fontproperties=my_font)
#
# plt.hist(t1,num_bins)
# plt.grid(True,alpha=0.5)
# plt.legend(prop=my_font,loc=0)
# plt.show()
# t2 = t2[t2[:,2]<=5000]
# plt.figure(figsize=(20,8),dpi=80)
# plt.xlabel("x轴",fontproperties=my_font)
# plt.ylabel("y轴",fontproperties=my_font)
# plt.title("数据显示",fontproperties=my_font)
# plt.scatter(t2[:,2],t2[:,-1])
# plt.grid(True,alpha=0.5)
# plt.legend(prop=my_font,loc=0)
# plt.show()

# zero_one = np.zeros((t1.shape[0],1)).astype(int)
# ones_one = np.ones((t2.shape[0],1)).astype(int)
# t1 = np.hstack((t1,zero_one))
# t2 = np.hstack((t2,ones_one))
#
# final = np.vstack((t1,t2))
#
# print(final)



#
# print(t2)
# print("*"*50)
# print(t2[:,3])

# a = np.array(range(30))
# b = a.reshape(6,5)
# print(b)
# c = np.array(range(10))
# d = c.reshape(2,5)
# print("*"*50)
# print(d)
# d = np.vstack(b)
# print(d)

# t = np.array([[  0.,   1.,   2.,   3.,   4.,   5.],
#        [  6.,   7.,  nan,   9.,  10.,  11.],
#        [ 12.,  13.,  14.,  nan,  16.,  17.],
#        [ 18.,  19.,  20.,  21.,  22.,  23.]])
# # num = np.count_nonzero(t!=t)
# # num_all = t.shape[0]*t.shape[1]
# # sum_not = t[np.isnan(t)==False].sum()
# t[t!=t] = t[np.isnan(t)==False].sum()/(t.shape[0]*t.shape[1]-np.count_nonzero(t!=t))
# print(t)

# import pandas as pd


# df = pd.read_csv("./dogNames2.csv")
# df = pd.Series(df)
# df = pd.Series(range(12),index=list("abcdefghijkl"))
# print(type(df),df.dtype,type(df.values),type(df.index))
# print(df['c'])

# df = pd.DataFrame(df)
# print(df.sort_values(by="Count_AnimalName",ascending=False).head(5))
# print(df.info)
# print(df.describe())
# print(df.shape,df.ndim,df.dtypes,df.index,type(df.columns),df.values)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager

# file_path = "./IMDB-Movie-Data.csv"
# file_data = pd.read_csv(file_path)

# # print(file_data)
# data_input = file_data["Runtime (Minutes)"].values
# # print(data_input)
# num_bin = (data_input.max()-data_input.min())//5
# x_act = list(range(data_input.min(),data_input.max()))[::5]
# # x_tricks =
#
# my_font = font_manager.FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=15)
# plt.figure(figsize=(20,8),dpi=80)
# plt.xlabel("x轴",fontproperties=my_font)
# plt.ylabel("y轴",fontproperties=my_font)
# plt.title("数据分布情况",fontproperties=my_font)
# plt.grid(True,alpha=0.5)
# plt.legend(prop=my_font,loc=0)
# plt.hist(data_input,num_bin)
# plt.xticks(x_act)
# plt.show()


# temp = file_data["Genre"].str.split(",").tolist()
# print(temp)
# temp_temp = [i for j in temp for i in j]
# data_input = list(set(temp_temp))
# print(data_input)
# Array_data = pd.DataFrame(np.zeros((file_data.shape[0], len(data_input))),columns=data_input)
# print(Array_data)
# for i in range(file_data.shape[0]):
#     Array_data.loc[i,temp[i]] = 1

# print(Array_data.head(1))
# new_count = Array_data.sum(axis=0)
# new_count = new_count.sort_values()
# x = new_count.index
# plt.figure(figsize=(20,8),dpi=80)
# plt.bar(range(len(x)),new_count.values,width=0.5)
# plt.xticks(range(len(x)),x)
# plt.show()


# t1 = pd.DataFrame(np.arange(9).reshape(3,3),columns=['a','b','c'])
# print(t1)
#
# t2 = pd.DataFrame(np.zeros((2,2)),columns=['e','f'])
# print(t2)
#
# t3 = t1.merge(t2,left_on="a",right_on="e",how="inner")
# print(t3)

# file_path = "./starbucks_store_worldwide.csv"
# file_data = pd.read_csv(file_path)
# print(file_data.info())
# us_num = file_data.groupby("Country").count()["Brand"]["US"]
# cn_num = file_data.groupby("Country").count()["Brand"]["CN"]
# print(us_num)
# # print(file_data.groupby("Brand").sum())
file_path = "./911.csv"
file_data = pd.read_csv(file_path)

# print(file_data.head())
# print(file_data.info())

# file_proc = file_data["title"].str.split(":").tolist()
# # data_add = list([i[0] for i in file_proc])
# # # data_input = pd.DataFrame(np.zeros((file_data.shape[0],len(data_add))),columns=data_add)
# # # print(data_input)
# # # for cate in data_add:
# # #     data_input[cate][file_data["title"].str.contains(cate)] = 1
# #
# # # print(data_input.sum(axis=0))
# # file_data["cate"] = data_add
# # # print(file_data.head(5))
# # temp = file_data.groupby(by="cate").count()["title"]
# # print(temp)
#

file_proc = file_data["title"].str.split(":").tolist()
data_add = list([i[0] for i in file_proc])
# data_input = pd.DataFrame(np.zeros((file_data.shape[0],len(data_add))),columns=data_add)
file_data["cate"] = data_add

file_data["timeStamp"] = pd.to_datetime(file_data["timeStamp"],format="")
file_data.set_index("timeStamp",inplace=True)

for item,items in file_data.groupby(by="cate"):
    temp = items.resample("M").count()["title"]
    plt.plot(range(temp.shape[0]),temp,label=item)
    _x = temp.index.strftime("%Y%m%d")
    plt.xticks(range(temp.shape[0]),_x,rotation=45)

# print(file_data.shape[0],file_data.index)
plt.legend(loc="best")
plt.show()

