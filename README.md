# 任务一：基于机器学习的文本分类

## 一、实验概述

### 1.1 实验背景

本实验使用回归模型，结合多种特征提取方法和优化策略，实现对电影评论的情感分析。

### 1.2 实验目标

（1）利用NumPy分析不同特征、损失函数、学习率对最终文本分类性能的影响

（2）了解Bag-of-Word、N-gram文本特征表示、分类器、（随机）梯度下降等知识

## 二、实验设计与实现

### 2.1 数据预处理

加载文本数据和情感标签，将短语字段Phrase以列表格式存储，情感标签Sentiment（0-4）转化为整型。按8:2比例划分训练集、测试集。

```
# 加载数据
df = pd.read_csv(Config.DATA_PATH)
texts = df["Phrase"].tolist()
labels = df["Sentiment"].astype(int).values
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(texts,labels,test_size=Config.TEST_SIZE,random_state=Config.RANDOM_STATE)
```

### 2.2 配置参数

```
class Config:
    # 数据集的路径
    DATA_PATH = r"D:/000personal/dataset/sentiment-analysis-on-movie-reviews/train.csv"
    # 模型保存的路径
    SAVE_DIR = r"D:/VSWorkSpace/Python/git-task1/output"

    # 特征配置
    N_GRAM = 2  #  词袋的大小
    TEST_SIZE = 0.2  # 测试集所占比例
    RANDOM_STATE = 21  #  随机数种子

    # 模型配置
    LEARNING_RATE = [0.1,1,10]  # 学习率
    EPOCHS = 10  # 训练轮数
    LOSS_FUNCTIONS = ["cross_entropy","hinge","focal"]  # 损失函数
    STRATEGIES = ["mini","sgd","batch"]  # 优化策略
    NGRAM_RANGE = [2,3]  # N-gram范围
    MAX_FEATURES = 2000  # 最大特征数
    BATCH_SIZE = 256  # 默认batch大小
```

### 2.3 特征提取和选择

#### 2.3.1 Bag-of-Words模型

构建词汇表，筛选最高频的MAX_FEATURES个词，生成稀疏矩阵，先使用COO格式构建，后转换为CSR格式加速矩阵运算。

```
class Bag:
    def __init__(self,max_features=Config.MAX_FEATURES):
        self.vocab = {}
        self.max_features = max_features

    def fit(self, texts):
        word_counts = defaultdict(int)
        for text in texts:
            for word in text.lower().split():
                word_counts[word] += 1
        # 只保留最常见的max_features个词
        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted_words[:self.max_features])}
        return self

    def transform(self, texts):
        """生成稀疏特征矩阵（词频模式）"""
        if self.vocab is None:
            # 如果vocab为空，则抛出异常
            raise ValueError("必须先调用fit方法")

        rows, cols, data = [], [], []
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                if word in self.vocab:
                    # 添加行索引
                        rows.append(i)
                    # 添加列索引
                        cols.append(self.vocab[word])
                    # 添加数据（词频）
                        data.append(1)
        # 使用COO格式构建后转为CSR（更高效）
        return csr_matrix((data, (rows, cols)), shape=(len(texts), len(self.vocab)))

    def fit_transform(self, texts):
        """合并fit和transform"""
        self.fit(texts)
        return self.transform(texts)
```

#### 2.3.2 N-gram模型

构建词汇表，保留最高频的MAX_FEATURES个N-gram，将文本转换为特征矩阵。

```
class Ngram:
    def __init__(self, n, max_features=Config.MAX_FEATURES):
        self.n = n
        self.vocab = {}
        self.max_features = max_features

    def build_vocabulary(self, texts):
        word_counts = defaultdict(int)
        for text in texts:
            words = text.lower().split()
            for i in range(len(words) - self.n + 1):
                ngram = " ".join(words[i : i + self.n])
                word_counts[ngram] += 1

        sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])[:self.max_features]

        self.vocab = {ngram: idx for idx, (ngram,_) in enumerate(sorted_words)}
        return self

    def create_features(self, texts):
        if self.vocab is None:
            raise ValueError("必须先调用build_vocabulary方法")
        features = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for idx, text in enumerate(texts):
            words = text.lower().split()
            for i in range(len(words) - self.n + 1):
                ngram = " ".join(words[i : i + self.n])
                if ngram in self.vocab:
                    features[idx, self.vocab[ngram]] += 1
        return features
```

### 2.4 Softmax回归模型

代码Softmax类实现了一个多类别分类器，使用softmax函数将模型输出转换为概率分布。支持三种损失函数：Cross-Entropy Loss、Hinge Loss、Focal Loss；三种优化策略：随机梯度下降（sgd）、小批量梯度下降（mini）、批量梯度下降（batch）。同时记录训练损失和测试集准确率信息。

```
class Softmax:
    def __init__(self, num_features, num_classes):
        """
        参数:
            num_features: 特征数量
            num_classes: 类别数量
        """
        # 简单随机初始化
        self.W = np.random.randn(num_features, num_classes).astype(np.float32) * 0.01
        self.b = np.zeros(num_classes, dtype=np.float32)
        self.num_classes = num_classes

    def softmax(self, scores):
        """稳定的softmax实现"""
        scores = scores - np.max(scores, axis=1, keepdims=True)  # 防溢出
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X):
        """预测类别"""
        scores = X.dot(self.W) + self.b
        return scores.argmax(axis=1)

    def compute_accuracy(self, X, y):
        """计算准确率"""
        pred = self.predict(X)
        return np.mean(y == pred)

    def compute_loss(self, X, y, loss_fn):
        """计算损失"""
        scores = X.dot(self.W) + self.b
        probs = self.softmax(scores)

        if loss_fn == "cross_entropy":
            correct_logprobs = -np.log(probs[np.arange(len(y)), y] + 1e-8)
            return np.mean(correct_logprobs)
        elif loss_fn == "hinge":
            margins = np.maximum(0, scores - scores[np.arange(len(y)), y].reshape(-1, 1) + 1)
            margins[np.arange(len(y)), y] = 0
            return np.mean(margins.sum(axis=1))
        elif loss_fn == "focal":
            pt = probs[np.arange(len(y)), y]
            alpha, gamma = 0.25, 2.0
            return -np.mean(alpha * (1 - pt) ** gamma * np.log(pt + 1e-8))
        else:
            raise ValueError(f"未知的损失函数: {loss_fn}")

    def compute_gradients(self, X_batch, y_batch, loss_fn):
        """计算梯度和损失"""
        num_samples = len(X_batch)
        scores = X_batch.dot(self.W) + self.b
        probs = self.softmax(scores)

        # 计算梯度
        if loss_fn == "cross_entropy":
            dscores = probs.copy()
            dscores[np.arange(num_samples), y_batch] -= 1
            dscores /= num_samples
        elif loss_fn == "hinge":
            margins = np.maximum(0, scores - scores[np.arange(num_samples), y_batch].reshape(-1, 1) + 1)
            margins[np.arange(num_samples), y_batch] = 0
            dscores = (margins > 0).astype(float)
            dscores[np.arange(num_samples), y_batch] = -dscores.sum(axis=1)
            dscores /= num_samples
        elif loss_fn == "focal":
            pt = probs[np.arange(num_samples), y_batch]
            alpha, gamma = 0.25, 2.0
            focal_weight = alpha * (1 - pt) ** gamma * (gamma * pt * np.log(pt + 1e-8) + pt - 1) / (pt * (1 - pt))
            dscores = probs.copy()
            dscores[np.arange(num_samples), y_batch] -= 1
            dscores *= focal_weight.reshape(-1, 1)
            dscores /= num_samples

        # 计算权重和偏置的梯度
        dW = X_batch.T.dot(dscores)
        db = np.sum(dscores, axis=0)

        return dW, db, self.compute_loss(X_batch, y_batch, loss_fn)

    def train(self, X_train, y_train, X_test, y_test,
              learning_rate=0.1, epochs=10, batch_size=256, 
              loss_fn="cross_entropy", strategy="mini", verbose=True):
        """
        训练Softmax回归模型

        参数:
            X_train: 训练数据
            y_train: 训练标签
            X_test: 测试数据
            y_test: 测试标签
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批量大小
            loss_fn: 损失函数(cross_entropy, hinge, focal)
            strategy: 优化策略(mini, sgd, batch)
            verbose: 是否打印训练过程
        """
        num_samples = len(X_train)
        train_loss_history = []
        test_acc_history = []

        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0

            if strategy == "sgd":
                # 随机梯度下降
                for i in range(num_samples):
                    X_batch = X_shuffled[i:i+1]
                    y_batch = y_shuffled[i:i+1]
                    dW, db, batch_loss = self.compute_gradients(X_batch, y_batch, loss_fn)
                    self.W -= learning_rate * dW
                    self.b -= learning_rate * db
                    epoch_loss += batch_loss

            elif strategy == "mini":
                # 小批量梯度下降
                for i in range(0, num_samples, batch_size):
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]
                    dW, db, batch_loss = self.compute_gradients(X_batch, y_batch, loss_fn)
                    self.W -= learning_rate * dW
                    self.b -= learning_rate * db
                    epoch_loss += batch_loss * len(X_batch)

            elif strategy == "batch":
                # 批量梯度下降
                dW, db, epoch_loss = self.compute_gradients(X_shuffled, y_shuffled, loss_fn)
                self.W -= learning_rate * dW
                self.b -= learning_rate * db
                epoch_loss *= num_samples

            # 计算平均损失
            epoch_loss /= num_samples
            train_loss_history.append(epoch_loss)

            # 计算测试集准确率
            test_acc = self.compute_accuracy(X_test, y_test)
            test_acc_history.append(test_acc)

            # 打印训练进度
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Test Acc: {test_acc:.4f}")

        return {
            "train_loss": train_loss_history,
            "test_acc": test_acc_history
        }
```

## 三、实验结果

（1）特征提取方法对比：词袋模型表现较好，2-gram、3-gram模型表现相近，学习率为10时效果显著下降，导致梯度爆炸、参数震荡等问题。

（2）损失函数表现对比：cross_entropy和hinge函数效果相近，focal函数效果最差，该函数常用于解决类别极度不平衡问题，电影评论情感分析的多分类任务类别不平衡程度可能未达到其期望情况。

（3）优化策略对比：训练过程体现小批量梯度下降的Loss下降更平滑，测试集准确率也较高，且批量大小为256，既保留了部分随机性，又降低了梯度方差。

（4）词序列长度对比：2-gram在准确率上优于3-gram，在收集情感信号时泛化能力更好。

由于设备限制，只设置了10轮训练次数，3种学习率（较大），进一步优化超参数可能提升模型效果。

![results_comparison](D:\VSWorkSpace\Python\git-task1\output\results_comparison.png)


