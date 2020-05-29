## 3.1 模型实用技巧
### 3.1.1 特征提升
#### 3.1.1.1 特征抽取
原始数据的种类有很多， 除了数字化的信号数据（声闻、图像）， 还有大量***符号化的文本***
。由于我们无法直接将符号化的文字本身用于计算任务， 所以需要通过某些处理手段， **预先将文本量化为特征向量**。
我们使用```DictVectorizer```对特征进行抽取和向量化。
```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangxing
# datetime:2020/5/29 2:39 下午
# software: PyCharm
from sklearn.feature_extraction import DictVectorizer

measurements = [
    {
        'city': 'Dubai',
        'temperature': 33.
    },
    {
        'city': 'London',
        'temperature': 12.
    },
    {
        'city': 'San Fransisco',
        'temperature': 18.
    }
]
vec = DictVectorizer()
data = vec.fit_transform(measurements)
data2 = vec.fit_transform(measurements).toarray()
print(data)
print(data2)
print(vec.get_feature_names())
```
```
  (0, 0)	1.0
  (0, 3)	33.0
  (1, 1)	1.0
  (1, 3)	12.0
  (2, 2)	1.0
  (2, 3)	18.0
[[ 1.  0.  0. 33.]
 [ 0.  1.  0. 12.]
 [ 0.  0.  1. 18.]]
['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']
```
对于文本数据：词袋法（Bag of Words）
将每条文本在高维的词表（Vocabulary）上映射出一个特征向量。 常见计算方式有两种：```CountVectorizer```和```TfidfVectorizer```.

|||
|---|---|
|```CountVectorizer```|考虑每种词汇（Term）在该条训练文本中出现的频率（Term Frequency）|
|```TfidfVectorizer```|考虑每种词汇（Term）在该条训练文本中出现的频率（Term Frequency）;考虑InverseDocumentFrequency|