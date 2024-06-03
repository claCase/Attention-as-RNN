# Attention-as-RNN
Non-Official Implementation of https://arxiv.org/pdf/2405.13956 implemented as a recurrent layer and efficient prefix-sum layer. 

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
N = 30
S = 20
for i in range(S):
    o = model(x, training=True)
    if i == S-1:
        plt.plot(o[N, :100, 0], color="blue", label="RNN-Attention")
    else:
        plt.plot(o[N, :100, 0], color="blue")

plt.plot(x[N, :100, 0], label="True")
plt.legend()
```
![](https://github.com/claCase/Attention-as-RNN/blob/main/figures/output_stochastic.png)