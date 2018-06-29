****
# 本模型是基于生成对抗网络的动画图片生成模型，本模型的模块：
|-checkpoints/ //保存中间的训练结果\
  |--netd_*.pth\
  |--netg_*.pth\
|-data //保存训练和测试数据\
 |-cantonFace //训练集\
|-img/train_back  //保存训练的中间结果每10个epoch保存一次\
|-img/generate/  //保存测试生成的图片\
|-main.py //包含训练和测试的代码\
|-model/model.py //定义GAN模型\
|-utils/visualize.py //使用visdom进行可视化\

***
# 使用的技术
- pythorch
- torchvison
- visdom

***

# 生成图片效果
https://https://github.com/lucineIT/GAN/blob/master/imgs/generate/generate_7_0.9243126154565551.png 生成图片

# 训练的中间结果
