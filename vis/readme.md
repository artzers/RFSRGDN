**安装依赖库**

在当前目录下，在终端执行

```bash
pip install -r requirements.txt
```

默认数据集文件夹结构：

```
./dataset/
    train/
        neg/
        	image1/
        	...
        pos/
        	image1/
        	...
    test/
        neg/
        	image1/
        	image2/
        	...
        pos/
            image1/
        	image2/
        	...
```

在**utils/config.py**中将**data_dir**改为自己数据集的路径

在终端中运行：

```bash
nohup python -m visdom.server &
```

在浏览器中打开：

http://localhost:8097/

------

下载预训练的模型

https://drive.google.com/open?id=15ILVDuwG14amJzedK7Mk5LkcJzDcATIr

**test**集上的结果F=0.999,数据集准备于2018-05-16

**依赖：**

**pytorch 0.4**

**python 3.6**