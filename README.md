# generateName
学习https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html 笔记：
【有感】神经网络具体训练过程通常可分为如下几步：预处理数据集（过滤脏数据等）、数据向量化（转成Tensor）、定义神经网络结构、训练和评估。因此，整理NLP FROM SCRATCH: GENERATING NAMES WITH A CHARACTER-LEVEL RNN代码时，尽可能按照上述四个步骤重构代码。

文件解释：
  preprocess.py：预处理数据集
  utils.py:工具类，包括wordToTensor、inputToTensor等；可添加自定义工具函数
  Net.py：神经网络类，为最简单的RNN；可自定义神经网络
  Main.py：训练 和 评估

运行方式：
  python xx.py
  eg：python Main.py
  0m 35s (5000 5%) 3.2158
  1m 8s (10000 10%) 2.6535
  1m 40s (15000 15%) 2.8754
  2m 13s (20000 20%) 2.4044
  2m 51s (25000 25%) 2.0412
  3m 25s (30000 30%) 2.4384
  3m 55s (35000 35%) 2.9582
  4m 30s (40000 40%) 2.1406
  5m 0s (45000 45%) 1.9513
  5m 30s (50000 50%) 2.2055
  6m 1s (55000 55%) 2.3472
  6m 30s (60000 60%) 2.1471
  7m 1s (65000 65%) 2.3621
  7m 30s (70000 70%) 2.4907
  8m 3s (75000 75%) 2.2371
  8m 36s (80000 80%) 2.5143
  9m 9s (85000 85%) 2.9792
  9m 39s (90000 90%) 2.2605
  10m 11s (95000 95%) 2.1944
  10m 41s (100000 100%) 1.8690

关于训练过程：代码中设置迭代次数为100000,；推荐不设置迭代次数，判断current_loss与pre_loss之间的差异小于给定值时，即可终止训练。
分别以C、H、I开头生成符合Chinese风格的名字：
  Chin
  Han
  Iun
分别以G、E、R开头生成符合German风格的名字：
  Gerten
  Erangen
  Rourter
