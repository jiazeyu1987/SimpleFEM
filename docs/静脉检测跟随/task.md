1.创建一个新的文件 auto_vein_detector.py
2.参考HEM模型下的屏幕捕获模式，先截取一个ROI2，这个ROI2的截取通过配置表来配置获取，类似simple_roi_daemon.py的ROI1的截取逻辑
3.ROI2的保存逻辑与simple_roi_daemon.py一样
4.获取到ROI2之后，选择阈值范围0~10内的，ROI2中心点所在的连通域作为mask，mask保存在ROI2的同一个目录下，mask和ROI的图片命名前缀一致
5.将ROI2移动到mask的中心点，大小不变
6.进行下一帧的检测（帧率可以在配置表里设置），检测到mask之后，保存ROI2,mask，将ROI2移动到mask中心点，继续下一帧检测
7.循环检测直到视频结束
8.使用单独的配置文件
9.日志保存在log文件夹下

