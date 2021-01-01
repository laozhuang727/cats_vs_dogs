# coding: utf-8

# In[1]:

import os
import shutil

# In[2]:
SMALL_DATASET_COUNT = 100

# == Step1 ==： 将文件type.num.jpg这样的名称，改为训练需要的格式（如：cat.0.jpg）

train_filenames = os.listdir('train')
train_filenames.sort()
train_cat = filter(lambda x: x[:3] == 'cat', train_filenames)
train_dog = filter(lambda x: x[:3] == 'dog', train_filenames)

test_filenames = os.listdir('test')
test_filenames.sort()
# test_filenames = filter(lambda x: int(x[:-4]) <= SMALL_DATASET_COUNT, test_file_dir)


# In[3]:

def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

# == Step2 ==：train，test文件夹准备
# 使用 Keras 的 ImageDataGenerator 需要将不同种类的图片分在不同的文件夹中。
# 因此我们需要对数据集进行预处理。这里我们采取的思路是创建符号链接(symbol link)

# rmrf_mkdir('train2')
# os.mkdir('train2/cat')
# os.mkdir('train2/dog')
#
# rmrf_mkdir('test2')
# os.symlink('../test/', 'test2/test')
#
#
# for filename in train_cat:
#     os.symlink('../../train/'+filename, 'train2/cat/'+filename)
#
# for filename in train_dog:
#     os.symlink('../../train/'+filename, 'train2/dog/'+filename)


# In[ ]:

# == Step3 ==：准备小型数据集，加速测试时间
rmrf_mkdir('train-small-dataset')
os.mkdir('train-small-dataset/cat')
os.mkdir('train-small-dataset/dog')

file_count = 0
for filename in train_cat:
    os.symlink('../../train/' + filename, 'train-small-dataset/cat/' + filename)
    file_count += 1
    if file_count >= SMALL_DATASET_COUNT:
        break

file_count = 0
for filename in train_dog:
    os.symlink('../../train/' + filename, 'train-small-dataset/dog/' + filename)
    file_count += 1
    if file_count >= SMALL_DATASET_COUNT:
        break

rmrf_mkdir('test-small-dataset')
rmrf_mkdir('test-small-dataset/test')

file_count = 0
for filename in test_filenames:
    os.symlink('../../test/' + filename, 'test-small-dataset/test/' + filename)
    file_count += 1
    if file_count >= SMALL_DATASET_COUNT:
        break
