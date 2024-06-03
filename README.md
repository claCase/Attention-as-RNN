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
plt.plot(o[30, :, 0], label="True")
plt.plot(x[30, :, 0], label="RNN-Attention")
plt.legend()

plt.plot(history.history["loss"])
plt.ylabel("Loss")
plt.xlabel("Epochs")
```

![](https://github.com/claCase/Attention-as-RNN/blob/main/figures/output.png)
![](https://github.com/claCase/Attention-as-RNN/blob/main/figures/loss_train.png)