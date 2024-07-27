# Attention-as-RNN
Non-Official Implementation of "Attention as an RNN" from https://arxiv.org/pdf/2405.13956 implemented as a recurrent layer and efficient prefix-sum layer. 

# Usage Example


```python
import numpy as np 
from src import models,layers,utils 
import matplotlib.pyplot as plt 

EPOCHS = 400
BATCH_SIZE = 50

# Init sin dataset 
x = utils.generate_sin()
x = tf.transpose(x, (2, 0, 1))

# Init Model
model = models.ScanRNNAttentionModel(heads=[10, 5], dims=[5, 2], activation="silu", concat_heads=False)
_ = model(x)
model.compile("adam", "mse")
# Train 
history = model.fit(x, x, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Visualize Result 
o = model(x)
plt.plot(o[30, :, 0], label="RNN-Attention")
plt.plot(x[30, :, 0], label="True")
plt.legend()

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
```

![](https://github.com/claCase/Attention-as-RNN/blob/main/figures/output.png)
![](https://github.com/claCase/Attention-as-RNN/blob/main/figures/loss_train.png)


You can also run the model in training mode, making the output stochasic via dropout and approximating the epistemic uncertainty 

```python
from src import plotter
fig, ax = plotter.plot_hist2d(scan_model, x[0], axis=0)
```

![](https://github.com/claCase/Attention-as-RNN/blob/main/figures/output_stochastic.png)

You can train the model using the pre-fix sum implementation and transfer the weights to the recurrent implementation to make inference recurrentely:
```python
config = model.get_config()
heads = config["heads"]
dims = config["dims"]
activation = config["activation"]
dropout = config["dropout"]
recurrent_dropout = config["recurrent_dropout"]

rnn_model = models.AttentionRNN(
    heads,
    dims,
    activation=activation,
    dropout=dropout,
    recurrent_dropout=recurrent_dropout,
)
rnn_model.build(x.shape)
rnn_model.set_weights(scan_model.get_weights())
rnn_model.compile("adam", "mse")
```

# Tensorboard Step-Time Graph 

![](https://github.com/claCase/Attention-as-RNN/blob/main/figures/step_time_graph.png)


# Time Series Classification 
```python
# Load dataset (!pip install aeon)
from aeon.datasets import load_classification
X, y = load_classification("ECG200")
y = y.astype(float)
y = np.where(y>0, 1., 0.)

plt.plot(X[:10, 0, :].T)
```
![](https://github.com/claCase/Attention-as-RNN/blob/main/figures/ecg200.png)

### ScanAttention Model (69% accuracy)
```python
# Construct model 
ki = tf.keras.Input(shape=(None, 1))
scan = models.ScanRNNAttentionModel([10, 10], [10, 10])
avg_pool = tf.keras.layers.GlobalAveragePooling1D()
max_pool = tf.keras.layers.GlobalMaxPooling1D()
min_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_min(x, -2))
conc = tf.keras.layers.Concatenate()
dense = tf.keras.layers.Dense(1, "sigmoid")

h = scan(ki)
avgp = avg_pool(h)
maxp = max_pool(h)
minp = min_pool(h)
mix = conc([avgp, maxp, minp])
o = dense(mix)

classification_model = tf.keras.Model(ki, o)
classification_model.compile("adam", "bce")

# Train
hist = classification_model.fit(X[:, 0, :], y, epochs=1000)

# Score
pred = classification_model.predict(X[:, 0, :])
tf.keras.metrics.Accuracy()(pred>0.5, y)
```

### LinearSelfAttention Model (74% accuracy)
```python
ki = tf.keras.Input(shape=(None, 1))

cnn = layers.LinearSelfAttention(5,10, "linear")
avg_pool = tf.keras.layers.GlobalAveragePooling1D()
max_pool = tf.keras.layers.GlobalMaxPooling1D()
min_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_min(x, -2))
conc = tf.keras.layers.Concatenate()
dense = tf.keras.layers.Dense(1, "sigmoid")

h = cnn(ki)
avgp = avg_pool(h)
maxp = max_pool(h)
minp = min_pool(h)
mix = conc([avgp, maxp, minp])
o = dense(mix)

classification_model_linear = tf.keras.Model(ki, o)
classification_model_linear.compile("adam", "bce")
```

### CNN1D Model (97% accuracy)
```python
ki = tf.keras.Input(shape=(None, 1))

cnn = tf.keras.layers.Conv1D(32,10)
avg_pool = tf.keras.layers.GlobalAveragePooling1D()
max_pool = tf.keras.layers.GlobalMaxPooling1D()
min_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_min(x, -2))
conc = tf.keras.layers.Concatenate()
dense = tf.keras.layers.Dense(1, "sigmoid")

h = cnn(ki)
avgp = avg_pool(h)
maxp = max_pool(h)
minp = min_pool(h)
mix = conc([avgp, maxp, minp])
o = dense(mix)

classification_model_cnn = tf.keras.Model(ki, o)
classification_model_cnn.compile("adam", "bce")
```

