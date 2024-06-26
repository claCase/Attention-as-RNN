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
fig, ax = plotter.plot_hist2d(scan_model, x[:1])
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
