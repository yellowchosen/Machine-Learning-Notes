# 1.0 Multiple Features
In this lecture, we would like introduce some new definitions and scenario. For example, multiple features for predicting house prices -- sizes, years, floors.

Also introduce new symbols：
* $n$: total numbers of features
* $x_{(j)}$: $j^{th}$ features
* $x^{(i)}$: features of $i^{th}$ training example
* $x_j^{(i)}$: value of feature $j$ in $i^{th}$ training example

By comparing simple regression model and multiple regression model:
$$ f_{w, b}(x) = wx+b $$
and for now
$$ f_{w, b}(x) = w_1x_1 + w_2x_2 + w_3x_3 +...+ w_nx_n + b $$

$\overrightarrow{w}^T = [w_0, w_1, w_2,..., w_n]$ <br>
b is a number <br>
$\overrightarrow{x}^T = [x_0, x_1, x_2,..., x_n]$ <br>

So recall the linear algebra we can simplify the formula to:
$$ f_{w, b}(x) = [w_0, w_1, w_2,..., w_n] * [x_0, x_1, x_2,..., x_n]^T = w^Tx $$


$$
\left[
\begin{matrix}
    x_0  \\\\
    x_1  \\\\
    x_2  \\\\
    .    \\\\
    .    \\\\
    .    \\\\
    x_n
\end{matrix}
\right] = x = [x_0, x_1, x_2,..., x_n]^T
$$

* $w^T$: transpose of $w$ matrix
* $x$: the features vector, we have $n+1$ dimensions features
* $x_0$: suppose $x_0^{(i)} = 1$

# 2.0 Appendix
Basic logic:
```python
w = np.array([1, 2, 3]) # w[0], w[1], w[2]
b = 4
x = np.array([10, 20, 30]) # x[0], x[1], x[2]
```
Without vectorization:
<img width="399" alt="Screen Shot 2022-07-11 at 10 11 01 PM" src="https://user-images.githubusercontent.com/99445916/178393533-7f3dc165-0c47-411b-9962-994fc8440bde.png">

```python
f = 0
for j in range(0, n):
    f = f + w[j] * x[j]
f = f + b
```

Vectorization(the most fast way!):
```python
f = np.dot(w, x) + b
```
