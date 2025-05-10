import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.dataloader import AV_KS_Dataset, AV_KS_Dataset_sample_level
from models.models import AVClassifier
import random

sample_nums = []

'''
设置随机种子等设定
'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



### key parameters:
## --warmup: warm up epochs. Default 5 epochs.

'''
设置参数
'''
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="KineticSound")
    parser.add_argument("--model", default="resnet18", type=str, choices=["resnet18"])
    parser.add_argument(
        "--modulation", default="sample", type=str, choices=["sample", "modality"]
    )
    parser.add_argument("--compare", default="none", type=str, choices=["none"])
    parser.add_argument("--n_classes", default=31, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--loader", default=165, type=int)
    parser.add_argument("--warmup", default=5, type=int)
    parser.add_argument("--bottom", default=5, type=int)
    parser.add_argument(
        "--encoder_lr_decay", default=0.1, type=float, help="decay coefficient"
    )

    parser.add_argument("--optimizer", default="adam", type=str, choices=["sgd", "adam"])
    parser.add_argument(
        "--learning_rate", default=5e-05, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--lr_decay_step", default=30, type=int, help="where learning rate decays"
    )
    parser.add_argument(
        "--lr_decay_ratio", default=0.1, type=float, help="decay coefficient"
    )

    parser.add_argument("--train", action="store_true", help="turn on train mode")
    parser.add_argument(
        "--log_path",
        default="log_sample",
        type=str,
        help="path to save tensorboard logs",
    )

    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--gpu_ids", default="2, 3", type=str, help="GPU ids")

    return parser.parse_args()

'''
执行调制，也就是估计各个样本的贡献，然后据此重采样
'''
def execute_modulation(args, model, device, dataloader, log_name, epoch):
    n_classes = args.n_classes

    contribution = {}
    softmax = nn.Softmax(dim=1)
    # cona， conv 是使用一个 epoch（整个数据集）预测后 a, v 模态的平均贡献度
    cona = 0.0
    conv = 0.0

    with torch.no_grad():
        # 要使用 eval 模式获取分数，说明模型应该是在基线情况下预训练过了。main 中是先训练一个 epoch，再执行调制
        model.eval()

        for step, (image, spec, label, index) in tqdm(enumerate(dataloader)):
            image = image.to(device)
            spec = spec.to(device)
            # label = label.to(device)
            a, v, out = model(spec.float(), image.float())
            # v, a 两个模态的输出。因为想要获得只有某个模态输入时的预测结果 （out_v, out_a），所以丢弃另一个模态
            out_v = model.module.exec_drop(a, v, drop="audio")
            out_a = model.module.exec_drop(a, v, drop="visual")
            # 融合、v、a 的各个类别预测概率
            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)
            # 对一个 batch 中的所有输入的真实类别
            for i, item in enumerate(label):
                all = prediction[i].cpu().data.numpy()
                # 预测结果的下标
                index_all = np.argmax(all)
                v = pred_v[i].cpu().data.numpy()
                index_v = np.argmax(v)
                a = pred_a[i].cpu().data.numpy()
                index_a = np.argmax(a)
                # 预测对了，分数就是输入模态数量；不对则为0
                value_all = 0.0
                value_a = 0.0
                value_v = 0.0
                if index_all == label[i]:
                    value_all = 2.0
                if index_v == label[i]:
                    value_v = 1.0
                if index_a == label[i]:
                    value_a = 1.0
                # 根据计算最终贡献的公式得出
                contrib_a = (value_a + value_all - value_v) / 2.0
                contrib_v = (value_v + value_all - value_a) / 2.0
                cona += contrib_a
                conv += contrib_v

                contribution[int(index[i])] = (contrib_a, contrib_v)
    # cona， conv 是使用一个 epoch（整个数据集）预测后 a, v 模态的平均贡献度
    cona /= len(dataloader.dataset)
    conv /= len(dataloader.dataset)

    if not os.path.exists(args.log_path + "/" + log_name):
        os.mkdir(args.log_path + "/" + log_name)
    if not os.path.exists(args.log_path + "/" + log_name + "/contribution"):
        os.mkdir(args.log_path + "/" + log_name + "/contribution")
    np.save(
        args.log_path + "/" + log_name + "/contribution/" + str(epoch) + ".npy",
        contribution,
    )
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("now train epoch, cona and conv: ", cona, conv)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    train_dataloader = None
    # print(contribution.keys())
    '''
    这个返回 train_dataloader 是要干什么？为什么预热阶段不返回
    因为预热阶段是稍微训练下模型，使其有一定的预测能力
    预热阶段过后，就开始根据之前的预测结果，计算出各模态贡献，进行重采样了
    而重采样就需要对训练数据进行更改
    '''
    if epoch >= args.warmup - 1:
        train_dataset = AV_KS_Dataset_sample_level(
            mode="train", loader=args.loader, contribution=contribution
        )
        # TODO: 当内存不够时，修改 num_workers 和 pin_memory
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=False,
        )

    return cona, conv, train_dataloader

'''
训练一个 epoch
'''
def train_epoch(args, epoch, model, device, dataloader, optimizer):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    _loss = 0

    for step, (image, spec, label, index, drop) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        image = image.to(device)
        spec = spec.to(device)
        label = label.to(device)
        drop = drop.to(device)
        a, v, out = model(spec.float(), image.float(), drop)

        loss = criterion(out, label)
        loss.backward()

        optimizer.step()
        _loss += loss.item()

    sample_nums.append(len(dataloader.dataset))

    return _loss / len(dataloader)

'''
warmup 阶段训练一个 epoch
'''
def warmup_epoch(args, epoch, model, device, dataloader, optimizer):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Warm up ... ")

    _loss = 0

    for step, (image, spec, label, index) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        image = image.to(device)
        spec = spec.to(device)
        label = label.to(device)
        # warmup 阶段没有 drop 丢弃模态
        a, v, out = model(spec.float(), image.float())

        loss = criterion(out, label)
        loss.backward()

        optimizer.step()
        _loss += loss.item()

    return _loss / len(dataloader)


def valid(args, model, device, dataloader, epoch, log_name):
    softmax = nn.Softmax(dim=1)
    print('testing...')
    n_classes = args.n_classes

    cri = nn.CrossEntropyLoss()
    _loss = 0

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (image, spec, label, index) in tqdm(enumerate(dataloader)):
            image = image.to(device)
            spec = spec.to(device)
            label = label.to(device)
            a, v, out = model(spec.float(), image.float())

            prediction = softmax(out)
            loss = cri(out, label)
            _loss += loss.item()

            for i, item in enumerate(label):
                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0

    return sum(acc) / sum(num)

'''
真正执行训练、重采样、测试等等是在这里
'''
def main():
    cona_all = []
    conv_all = []
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device("cuda:0")
    # 送入编码器
    model = AVClassifier(args)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    # 准备数据集和 dataloader
    train_dataset = AV_KS_Dataset(mode="train", loader=args.loader)
    train_val_dataset = AV_KS_Dataset(mode="train", loader=args.loader)
    test_dataset = AV_KS_Dataset(mode="test", loader=args.loader)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    train_val_dataloader = DataLoader(
        train_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16
    )

    # 设置优化器和调度器
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4,
            amsgrad=False,
        )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, args.lr_decay_step, args.lr_decay_ratio
    )
    # 开始训练
    if args.train:
        best_acc = 0.0
        # 对每个 epoch
        for epoch in range(args.epochs):
            # 打印日志
            print("Epoch: {}: ".format(epoch))
            writer_path = os.path.join(args.log_path)
            if not os.path.exists(writer_path):
                os.mkdir(writer_path)
            log_name = "{}_{}_{}_{}_{}_epochs{}_batch{}_lr{}_en{}_warmup{}".format(
                args.compare,
                args.optimizer,
                args.dataset,
                args.modulation,
                args.model,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.encoder_lr_decay,
                args.warmup,
            )

            # warmup/正常情况下，训练一个 epoch
            if epoch < args.warmup:
                batch_loss = warmup_epoch(
                    args, epoch, model, device, train_dataloader, optimizer
                )
            else:
                batch_loss = train_epoch(
                    args, epoch, model, device, train_dataloader, optimizer
                )
            '''
            执行调制，得到 a, v 的单模态贡献 cona， conv
            非预热阶段，还会得到 train_dataloader
            因为预热阶段是稍微训练下模型，使其有一定的预测能力
            预热阶段过后，就开始根据之前的预测结果，计算出各模态贡献，进行重采样了
            而重采样就需要对训练数据进行更改
            '''
            if epoch >= args.warmup - 1:
                cona, conv, train_dataloader = execute_modulation(
                    args, model, device, train_val_dataloader, log_name, epoch
                )
            else:
                cona, conv, _ = execute_modulation(
                    args, model, device, train_val_dataloader, log_name, epoch
                )
            cona_all.append(cona)
            conv_all.append(conv)
            # sgd 走一步，获取 acc
            scheduler.step()
            acc = valid(args, model, device, test_dataloader, epoch, log_name)
            # 保存最佳模型和相关信息（如有）
            if acc > best_acc:
                best_acc = float(acc)

                model_name = "{}_best_model_of_{}_{}_{}_{}_epochs{}_batch{}_lr{}_en{}_warmup{}_bottom{}_{}_{}.pth".format(
                    args.compare,
                    args.optimizer,
                    args.dataset,
                    args.modulation,
                    args.model,
                    args.epochs,
                    args.batch_size,
                    args.learning_rate,
                    args.encoder_lr_decay,
                    args.warmup,
                    args.bottom,
                    args.dynamic,
                    args.method,
                )

                saved_dict = {
                    "saved_epoch": epoch,
                    "acc": acc,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }

                save_dir = os.path.join(args.log_path, model_name)

                torch.save(saved_dict, save_dir)
                print("The best model has been saved at {}.".format(save_dir))
                print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))
            # 否则只是打印当前 epoch 训练信息
            else:
                print(
                    "Loss: {:.4f}, Acc: {:.4f}, Best Acc: {:.4f}".format(
                        batch_loss, acc, best_acc
                    )
                )


if __name__ == "__main__":
    main()
