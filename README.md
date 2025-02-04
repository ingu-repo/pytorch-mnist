# Deep Learning
入力層やハイパーパラメータ、損失関数などの学習手順さえ教えてあげれば、
勝手に学習してくれるのがDLの素晴らしいところ。

# Virtual Env
Creating virtual env as a name of venv
```shell
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

# PyTorch
### Summary
逆伝播をするための勾配を値を計算してくれる

### Numpy vs PyTorch
Array vs Tensor

|Points|Numpy|PyTorch|
|---|---|---|
|Type Name|array|tensor|
|Default Type of 整数|int32|int64|
|Default Type of 実数|float64|float32|
|CPU|OK|OK|
|GPU|NG|OK|

### 損失関数 : 平均２乗法 (MSE)

```python
    criterion = nn.MSELoss()
    x = torch.Tensor([[0, 1, 2]])   # 純伝播の結果
    y = torch.Tensor([[1, -1, 0]])  # 正解ラベル
    loss = criterion(x, y)          # 損失値
```
$$
\frac{(x1-x2)^{2}+(y1-y2)^{2}+(z1-z2)^{2}}{3}
$$

### 最適化関数
最適化関数を宣言するにはモデルが宣言されている必要がある。
そのためテストにはすでにあるモデル(vgg)を使用
```python
    model = vgg.vgg11()
    optimizer = optim.Adam(model.parameters())
    print (optimizer)
```

### モデルの構築
|Item|Desc|
|---|---|
|__init__()|既定の名前
|forward()|既定の名前

### 損失関数 : クロスエントロピー

### PyTorch Notes
* default data type and size is different b/w Numpy and Torch
* matplotlib can handle only numpy data type, so needs to adjust torch type to numpy to show graph
* data in cpu can NOT calculate with data in gpu
* usually data come from Numpy so first run the data in CPU and need to send it to GPU
* y.backward() keeps x.grad values so in case loop, need initialization of grad

### tensor vs Tensor
* tensor : data type auto check such as int64 if it is number
* Tensor : use default type such as float32


# Apendix


## 学習率 (Learning Rate)
一回の更新でどれくらいパラメータを変化させるのかを決める

## 最適化
学習率を調整していくこと。<br>
Adamが代表的な手法。

## ハイパーパラメータ
人が決めるパラメータのこと。
* バッチサイズ（全てのデータ数ではなく、ミニバッチのサイズを指す）
* epoch数（学習のループ数: 全てのデータを全部流して１epoch）
* モデルの構造

### define-and-run 
* first make a model and learning with data
* fast to learn
* but hard to know 中間層 and hard to debug

### define-by-run
* starting together both making a model and learning
* slow to learn
* but easy to know 中間層 and easy to debug
* model can be fixed easily whereas define-and-run is difficult



## 単回帰分析 : 直線の傾き
$
a=\displaystyle\sum_{n=1}^{n} \frac{(x_n y_n)}{(x_n)^2}
$

### スカラー
$
a, b, x, y, etc 
$

### ベクトル
$
x=\left[
\begin{array}{c}
a_{1} \\
a_{2} \\
a_{3} \\
\end{array}
\right]
$

## 行列
$
X=\left[
\begin{array}{ccc}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{array}
\right]
$

### テンソル
$
行列の集合体
$

### 行列の足し算
$
条件：サイズが同じであること
$

## 行列のかけ算 : 行と列でかけ算、各かけ算の合計
$ 
\left[
\begin{array}{ccc}
1 & 2 \\
3 & 4 \\
\end{array}
\right]
\left[
\begin{array}{ccc}
5 & 6 \\
7 & 8 \\
\end{array}
\right]
=\left[
\begin{array}{ccc}
1*5+2*7 & 1*6+2*8 \\
3*5+4*7 & 8 \\
\end{array}
\right]
$

## 行列のかけ算 : 注意事項
$
A(m * n) B(o * p) = C
$

$
m * nとo * pが同じサイズである必要なし、ただしnとoは同じである必要あり。
$

$
Cはm * pのサイズで出来上がる性質がある。
$

### サイズ感
$　
横ベクトル　縦ベクトル　=　スカラー
$

$　
行列　縦ベクトル　=　縦ベクトル
$

$　
横ベクトル 行列　縦ベクトル　=　スカラー
$

$　
単位行列　I ：　かけ算しても変わらない。行列　X　単位行列　I　=　行列　X
$

$
単位行列　I=\left[
\begin{array}{ccc}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{array}
\right]
$

$　
逆行列　：　かけ算して１になる行列　：　X^{-1}
$

### 行列で微分
$
２つの縦行列Aとxがある場合、Aを横にした行列とxのかけ算をxで微分すると縦行列Aになる
$

$
\frac{δ}{δx}(A^t x) = A
$

