# Ranger25

**Overview**:

`Ranger25` is an experimental composite optimizer that blends together seven advanced optimization techniques—ADOPT, AdEMAMix, Cautious updates, StableAdamW/Adam‑atan2, OrthoGrad, adaptive gradient clipping (AGC), and Lookahead—to achieve more reliable convergence, improved stability, and faster training across a wide range of deep‑learning tasks. By combining theoretical convergence fixes (ADOPT) with enhanced utilization of past gradients (AdEMAMix), directional masking (Cautious), numerical stability (Adam‑atan2), gradient decorrelation (OrthoGrad), unit‑wise clipping (AGC), and periodic weight averaging (Lookahead), Ranger25 aims to deliver the best of each world in a single optimizer.

**Parameters**:

- **`learning_rate`** *(float, default=1e-3)*: Base step size for parameter updates.
- **`betas`** *(tuple of three floats, default=(0.9, 0.98, 0.9999))*:
  * `beta1` for first‑moment EMA (momentum)
  * `beta2` for second‑moment EMA (RMS scaling)
  * `beta3` for slow EMA used in the “mix” component (AdEMAMix).
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominator (and in StableAdamW/Adam‑atan2 branch).
- **`weight_decay`** *(float, default=1e-3)*: Coefficient for decoupled weight‑decay regularization (AdamW style).
- **`alpha`** *(float, default=5.0)*: Mixing coefficient magnitude for the slow EMA in AdEMAMix.
- **`t_alpha_beta3`** *(int or None, default=None)*: Number of steps over which to warm up `alpha` and `beta3`; if `None`, no warmup.
- **`lookahead_merge_time`** *(int, default=5)*: Number of steps between Lookahead slow‑weight merges.
- **`lookahead_blending_alpha`** *(float, default=0.5)*: Interpolation factor between fast and slow weights at each Lookahead merge.
- **`cautious`** *(bool, default=True)*: Enable Cautious updates—masking out parameter updates whose sign conflicts with the raw gradient.
- **`stable_adamw`** *(bool, default=True)*: Use StableAdamW variant, which rescales step size by measured gradient variance for numerical stability.
- **`orthograd`** *(bool, default=True)*: Enable OrthoGrad, projecting each gradient to be orthogonal to its parameter vector before update.
- **`weight_decouple`** *(bool, default=True)*: Apply weight decay in a decoupled fashion (AdamW) rather than via loss augmentation.
- **`fixed_decay`** *(bool, default=False)*: Use fixed weight‑decay (not scaled by learning rate) when `weight_decouple` is True.
- **`clipnorm`** *(float or None)*: Clip gradients by global L‑2 norm.
- **`clipvalue`** *(float or None)*: Clip gradients by value.
- **`global_clipnorm`** *(float or None)*: Alias for clipping by global norm across all parameters.
- **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average of model weights.
- **`ema_momentum`** *(float, default=0.99)*: Decay rate for weight EMA.
- **`ema_overwrite_frequency`** *(int or None)*: How often to overwrite model weights with EMA weights.
- **`loss_scale_factor`** *(float or None)*: Static loss‑scaling factor for mixed‑precision training.
- **`gradient_accumulation_steps`** *(int or None)*: Number of steps to accumulate gradients before applying an update.
- **`name`** *(str, default="ranger25")*: Name identifier for the optimizer instance.

**Example Usage**:

```python
import tensorflow as tf
from ranger25 import Ranger25

# Instantiate the Ranger25 optimizer with custom settings
optimizer = Ranger25(
    learning_rate=3e-4,
    betas=(0.9, 0.98, 0.9999),
    epsilon=1e-8,
    weight_decay=1e-4,
    alpha=4.0,
    t_alpha_beta3=10000,
    lookahead_merge_time=6,
    lookahead_blending_alpha=0.6,
    cautious=True,
    stable_adamw=True,
    orthograd=True,
    fixed_decay=False,
    clipnorm=1.0,
    use_ema=True,
    ema_momentum=0.995,
    gradient_accumulation_steps=2,
    name="ranger25_custom"
)

# Compile a Keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=20)
```
