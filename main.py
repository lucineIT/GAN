import os
import ipdb
import torch as t
import torchvision as tv
import tqdm
import random
from model.model import NetGenerator, NetD
from torch.autograd import Variable
from torchnet.meter import AverageValueMeter
#gui
import tkinter as tk
from PIL import Image, ImageTk

class Config(object):
    data_path='data/'
    num_workers=4
    image_size=96
    batch_size=256
    max_epoch=200
    G_lr=2e-4
    D_lr=2e-4
    beta1=0.5
    use_gpu=True
    nz=100
    ngf=64
    ndf=64

    save_path='imgs/train_backup/'

    vis=True
    env='py36'
    plot_every=20

    debug_file='/tmp/debuggan'
    d_every=1
    g_every=5
    decay_every=10
    netd_path='checkpoints/netd_1920.pth'
    netg_path='checkpoints/netg_1920.pth'
    startpoint=1920
    gen_img='imgs/generate/'

    gen_num=64
    gen_search_num=512
    gen_mean=0
    gen_std=1



def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    if opt.vis:
        from utils.visualize import Visualizer
        vis = Visualizer(opt.env)

    transforms = tv.transforms.Compose([
        tv.transforms.Scale(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True
                                         )

    # 定义网络
    netg, netd = NetGenerator(opt), NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.G_lr, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.D_lr, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss()

    # 真图片label为1，假图片label为0
    # noises为生成网络的输入
    true_labels = Variable(t.ones(opt.batch_size))
    fake_labels = Variable(t.zeros(opt.batch_size))
    fix_noises = Variable(t.randn(opt.batch_size, opt.nz, 1, 1))
    noises = Variable(t.randn(opt.batch_size, opt.nz, 1, 1))

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    if opt.use_gpu:
        netd.cuda()
        netg.cuda()
        criterion.cuda()
        true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
        fix_noises, noises = fix_noises.cuda(), noises.cuda()

    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = Variable(img)
            if opt.use_gpu:
                real_img = real_img.cuda()
            if ii % opt.d_every == 0:
                # 训练判别器
                optimizer_d.zero_grad()
                ## 尽可能的把真图片判别为正确
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                ## 尽可能把假图片判别为错误
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()  # 根据噪声生成假图
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.data[0])

            if ii % opt.g_every == 0:
                # 训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.data[0])

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                ## 可视化
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.data.cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.plot('error_d', errord_meter.value()[0])
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('error_g', errorg_meter.value()[0])


        if epoch % opt.decay_every == 0:
            # 保存模型、图片
            tv.utils.save_image(fix_fake_imgs.data[:64], '%s/%s.png' % (opt.save_path, (epoch+opt.startpoint)), normalize=True,
                                range=(-1, 1))
            t.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % (epoch+opt.startpoint))
            t.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % (epoch+opt.startpoint))
            errord_meter.reset()
            errorg_meter.reset()
            optimizer_g = t.optim.Adam(netg.parameters(), opt.G_lr, betas=(opt.beta1, 0.999))
            optimizer_d = t.optim.Adam(netd.parameters(), opt.D_lr, betas=(opt.beta1, 0.999))


def generate(image_index,**kwargs):
    """
    随机生成动漫头像，并根据netd的分数选择较好的
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    netg, netd = NetGenerator(opt).eval(), NetD(opt).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = Variable(noises, volatile=True)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    if opt.use_gpu:
        netd.cuda()
        netg.cuda()
        noises = noises.cuda()

    # 生成图片，并计算图片在判别器的分数
    fake_img = netg(noises)
    scores = netd(fake_img).data

    # 挑选最好的某几张
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])
    # 保存图片
    tv.utils.save_image(t.stack(result), opt.gen_img+"generate_%s_%s.png" % (str(image_index), str(random.random())), normalize=True, range=(-1, 1))
    return result

if __name__ == '__main__':
    opt = Config()
    # window = tk.Tk()
    # window.title('动画生成器')
    # window.geometry('1024x7680')
    #
    # # welcome image
    # canvas = tk.Canvas(window, height=1024, width=768)
    # wifi_img = Image.open('imgs/train_backup/1000.png')
    # image_file = ImageTk.PhotoImage(wifi_img)
    #
    # image = canvas.create_image(0, 0, anchor='nw', image=image_file)
    # canvas.pack(side='top')
    #
    #
    # def gen():
    #     wifi_img=generate(1)
    #     return wifi_img
    # # control button
    # btn_login = tk.Button(window, text='训练', command=train)
    # btn_login.place(x=10, y=20)
    # #txtbx=tk.Text(window)
    #
    # #txtbx.place(x=10, y=60)
    # #txtbx.insert(0, "训练参数")
    # btn_sign_up = tk.Button(window, text='测试', command=gen)
    # btn_sign_up.place(x=10, y=50)
    # # img_png = ImageTk.PhotoImage(wifi_img)
    # # img_label = tk.Label(window, img_png)
    # # img_label.pack()
    # window.mainloop()
    import fire

    #训练时时使用下面这段代码
    #train()
    #训练结束

    #测试生成图片时使用下面这段代码

    image_number=50
    for i in range(image_number):
        generate(i)
    #测试结束

    #GUI

    fire.Fire()
