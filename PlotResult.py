  xpt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    Data_1, = plt.plot(xpt, one_layer_gnet_acc_test, label='Test')
    Data_2, = plt.plot(xpt, one_layer_gnet_acc_train, label='Train')
    plt.title("Dogs Vs Cats")  # 圖表標題
    plt.xlabel("epoch")  # x軸標題
    plt.ylabel("accuracy")  # y軸標題
    plt.legend(handles=[Data_1, Data_2])
    plt.show()  # 顯示繪製的圖形
    with open("../metrics.json）
