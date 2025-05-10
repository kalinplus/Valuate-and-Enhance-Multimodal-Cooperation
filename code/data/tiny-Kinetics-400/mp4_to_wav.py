import os
import csv

train_videos = './tiny-Kinetics-400/annotations/tiny_train.csv'
test_videos = './tiny-Kinetics-400/annotations/tiny_val.csv'

train_audio_dir = './tiny-Kinetics-400-audio/train-set'
test_audio_dir = './tiny-Kinetics-400-audio/test-set'

'''
数据长这样：
label,youtube_id,time_start,time_end,split,is_cc
abseiling,-3B32lodo2M,59,69,val,0
"air drumming",_dbcJuKJQNs,40,50,val,0
'''

# test set processing
with open(test_videos, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    rows = list(reader)  # reader 没有 len()，转换成列表
    for i, row in enumerate(rows):
        if i % 500 == 0:
            print('*******************************************')
            print('{}/{}'.format(i, len(rows)))
            print('*******************************************')
        category = row[0].replace(' ', '_')
        filename = row[1] + '_' + row[2].zfill(6) + '_' + row[3].zfill(6)
        mp4_filename = os.path.join('.\\tiny-Kinetics-400-data\\', category, filename + '.mp4')
        wav_filename = os.path.join('.\\tiny-Kinetics-400-audio\\test-set', filename + '.wav')
        # ! 与 process.audio.py 期望的目录结构搭配，所有文件都保存在 test-set 下
        # ffmpeg 不支持创建多级目录/文件，想要保持原有目录结构就先创建
        # category_path = os.path.join('.\\tiny-Kinetics-400-audio\\test-set', category)
        # if not os.path.exists(category_path):
        #     os.makedirs(category_path)
        if os.path.exists(wav_filename):
            pass
        else:
            os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))

# train set processing
with open(train_videos, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过标题行
    rows = list(reader)
    for i, row in enumerate(rows):
        if i % 500 == 0:
            print('*******************************************')
            print('{}/{}'.format(i, len(rows)))
            print('*******************************************')
        category = row[0].replace(' ', '_')
        filename = row[1] + '_' + row[2].zfill(6) + '_' + row[3].zfill(6)
        mp4_filename = os.path.join('.\\tiny-Kinetics-400-data\\', category, filename + '.mp4')
        wav_filename = os.path.join('.\\tiny-Kinetics-400-audio\\train-set', category, filename + '.wav')
        # ! 与 process.audio.py 期望的目录结构搭配，所有文件都保存在 train-set 下
        # category_path = os.path.join('.\\tiny-Kinetics-400-audio\\train-set', category)
        # if not os.path.exists(category_path):
        #     os.makedirs(category_path)
        if os.path.exists(wav_filename):
            pass
        else:
            os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))





