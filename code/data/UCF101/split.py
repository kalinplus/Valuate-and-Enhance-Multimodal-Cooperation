import shutil,os

txtlist = ['./ucfTrainTestlist/testlist01.txt']
dataset_dir = './UCF-101/'   #数据存放路径
copy_path = './val/'         #验证集存放路径，把原本测试集当成验证集了，记得创建目录
# 原本的 UCF101 数据集 变为 训练集

for txtfile in txtlist:
	for line in open(txtfile, 'r'):
		# 源文件
		o_filename = dataset_dir + line.strip()
		# 目标路径
		n_filename = copy_path + line.strip()
		if not os.path.exists('/'.join(n_filename.split('/')[:-1])):
			os.makedirs('/'.join(n_filename.split('/')[:-1]))
		# 移动文件
		shutil.move(o_filename, n_filename)
