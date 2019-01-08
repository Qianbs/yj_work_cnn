# yj_work_cnn
利用卷积神经网络进行手写数字识别



1，主要流程说明：

a,使用tensorflow构建基于LeNet-5模型的cnn网络，对mnist数据集进行训练2万次，保存训练好的模型参数文件；

b,使用python语言编写数字识别监听程序，该程序基于tensorflow加载训练好的模型参数文件，同事开启一个tcp连接监听，将接收到的图片文件数据输入cnn网络进行计算，预测数字识别结果；

c，使用java web应用提供一个手写输入界面，将用户的手写输入实时转换为图片，并发送给上述数字识别程序，获取识别结果后再发送给前端页面显示。



2，运行环境：linux、python3.5、tensorflow、tomcat。



3，运行步骤：
	
a,切换到目录：cd cnn_number；
	
b,执行预测程序：./predict.sh；
	
c，拷贝web目录下的number.war到tomcat的webapp目录，启动tomcat；
	
d，在浏览器中访问http://127.0.0.1:8080/number；
	
e，在页面上通过鼠标手写数字，查看数字识别结果。



4，cnn_number:实现基于mnist数据集进行cnn训练和预测。

其中：MNIST_data为mnist数据集；
mnist_variables为训练好的模型参数文件；
train.py为训练程序；
predict.py为预测程序；
predict.sh为启动预测程序的脚本。



5，web目录实现手写输入界面、手写图片保存等功能。

其中：number目录为基于java的web应用源码；

number.war为web应用包，可直接放到tomcat的webapp中，随tomcat运行。

主要的代码文件：

number.html为前端页面，提供手写输入界面，保存手写输入图片并传送到后台；

NumServlet.java为Web应用后端java程序，处理前端传递的图片，进行格式转换、缩放、灰度化、中心化等处理，然后通知python的预测程序进行对约定路径(/root/temp,需要有该目录的操作权限)的图片进行预测，并将结果返回给前端。
