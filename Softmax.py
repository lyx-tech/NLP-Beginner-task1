from collections import defaultdict
import itertools
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
import string

class Config:
    # 数据集的路径
    DATA_PATH = r"D:/000personal/dataset/sentiment-analysis-on-movie-reviews/train.csv"
    # 模型保存的路径
    SAVE_DIR = r"output"

    # 特征配置
    N_GRAM = 2  #  词袋的大小
    TEST_SIZE = 0.2  # 测试集所占比例
    RANDOM_STATE = 21  #  随机数种子

    # 模型配置
    LEARNING_RATE = [0.1,1,10]  # 学习率
    EPOCHS = 10  # 训练轮数
    LOSS_FUNCTIONS = ["cross_entropy"]  # 损失函数
    STRATEGIES = ["mini","sgd","batch"]  # 优化策略
    NGRAM_RANGE = [2,3]  # N-gram范围
    MAX_FEATURES = 2000  # 最大特征数
    BATCH_SIZE = 256  # 默认batch大小


# 创建保存目录
os.makedirs(Config.SAVE_DIR, exist_ok=True)

# 加载文本和标签
def load_data(data_path=Config.DATA_PATH):
    df = pd.read_csv(Config.DATA_PATH, sep=",", encoding="utf-8")
    return df["Phrase"].to_list(), df["Sentiment"].astype(int).values

# 特征提取
# Bag-of-Word
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

def visualize_training(history):
    """可视化训练过程"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_results(results_df):
    """可视化结果比较"""
    plt.figure(figsize=(18, 12))
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", 8)
    
    # 1. 不同特征提取方法对比
    plt.subplot(2, 2, 1)
    for i, feat in enumerate(results_df['feature'].unique()):
        subset = results_df[results_df['feature'] == feat]
        sns.lineplot(data=subset, x='learning_rate', y='test_acc', 
                     label=feat, color=palette[i], marker='o')
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Test Accuracy")
    plt.title("Feature Extraction Methods Comparison")
    plt.legend()
    
    # 2. 不同损失函数对比
    plt.subplot(2, 2, 2)
    for i, loss in enumerate(Config.LOSS_FUNCTIONS):
        subset = results_df[results_df['loss'] == loss]
        sns.lineplot(data=subset, x='learning_rate', y='test_acc',
                     label=loss, color=palette[i], marker='s')
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Test Accuracy")
    plt.title("Loss Functions Comparison")
    plt.legend()
    
    # 3. 不同优化策略对比
    plt.subplot(2, 2, 3)
    for i, strategy in enumerate(Config.STRATEGIES):
        subset = results_df[results_df['strategy'] == strategy]
        sns.lineplot(data=subset, x='learning_rate', y='test_acc',
                     label=strategy, color=palette[i+3], marker='D')
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Test Accuracy")
    plt.title("Optimization Strategies Comparison")
    plt.legend()
    
    # 4. N-gram大小对比
    plt.subplot(2, 2, 4)
    ngram_df = results_df[results_df['feature'].str.contains('gram')]
    for i, n in enumerate(Config.NGRAM_RANGE):
        subset = ngram_df[ngram_df['feature'] == f"{n}-gram"]
        sns.lineplot(data=subset, x='learning_rate', y='test_acc',
                     label=f"{n}-gram", color=palette[i+6], marker='^')
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Test Accuracy")
    plt.title("N-gram Size Comparison")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAVE_DIR, 'results_comparison.png'), dpi=300)
    plt.show()

def main():
    # 创建保存目录
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    # 加载数据
    df = pd.read_csv(Config.DATA_PATH)
    texts = df["Phrase"].tolist()
    labels = df["Sentiment"].astype(int).values
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    # 初始化结果列表
    results = []
    
    # 特征提取方法测试
    for feat_name, vectorizer in [
        ('Bag-of-Words', Bag(max_features=Config.MAX_FEATURES)),
        *[(f'{n}-gram', Ngram(n=n, max_features=Config.MAX_FEATURES)) for n in Config.NGRAM_RANGE]
    ]:
        # 特征转换
        print(f"\nProcessing features: {feat_name}")
        
        if isinstance(vectorizer, Bag):
            X_train_feat = vectorizer.fit_transform(X_train).toarray()
            X_test_feat = vectorizer.transform(X_test).toarray()
        else:
            vectorizer.build_vocabulary(X_train)
            X_train_feat = vectorizer.create_features(X_train)
            X_test_feat = vectorizer.create_features(X_test)
        
        # 参数组合
        for lr, loss_fn, strategy in itertools.product(
            Config.LEARNING_RATE, Config.LOSS_FUNCTIONS, Config.STRATEGIES
        ):
            print(f"\nTraining with: lr={lr}, loss={loss_fn}, strategy={strategy}")
            
            # 初始化模型
            model = Softmax(
                num_features=X_train_feat.shape[1],
                num_classes=5
            )
            
            # 训练模型
            history = model.train(
                X_train_feat, y_train,
                X_test_feat, y_test,
                learning_rate=lr,
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                loss_fn=loss_fn,
                strategy=strategy
            )
            
            # 可视化训练过程
            visualize_training(history)
            
            # 最终评估
            train_acc = model.compute_accuracy(X_train_feat, y_train)
            test_acc = model.compute_accuracy(X_test_feat, y_test)
            
            print(f"Final Results - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            # 保存结果
            results.append({
                "feature": feat_name,
                "learning_rate": lr,
                "loss": loss_fn,
                "strategy": strategy,
                "train_acc": train_acc,
                "test_acc": test_acc
            })
    
    # 保存和可视化结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(Config.SAVE_DIR, "softmax_results.csv"), index=False)
    visualize_results(results_df)
    
    # 打印最佳组合
    best_idx = results_df["test_acc"].idxmax()
    print("\nBest Combination:")
    print(results_df.loc[best_idx])

if __name__ == "__main__":
    main()

'''
Best Combination:
feature           Bag-of-Words
learning_rate              1.0
loss             cross_entropy
strategy                  mini
train_acc             0.605072
test_acc              0.587851
Name: 3, dtype: object
'''