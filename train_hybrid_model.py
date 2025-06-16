from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape, Activation
from tensorflow.keras.optimizers import Adam

def hybrid_model(input_shape=(66, 200, 3), transformer_dim=64, num_heads=4, ff_dim=128, dropout_rate=0.1):
    inputs = Input(shape=input_shape)

    # CNN layers same as NVIDIA but using functional API
    x = Conv2D(24, (5,5), strides=(2,2), activation='elu')(inputs)
    x = Conv2D(36, (5,5), strides=(2,2), activation='elu')(x)
    x = Conv2D(48, (5,5), strides=(2,2), activation='elu')(x)
    x = Conv2D(64, (3,3), activation='elu')(x)
    x = Conv2D(64, (3,3), activation='elu')(x)

    # Flatten CNN output
    x = Flatten()(x)

    # Project CNN features to transformer_dim
    x = Dense(transformer_dim, activation='relu')(x)

    # Add sequence dimension: (batch_size, sequence_length=1, transformer_dim)
    x = Reshape((1, transformer_dim))(x)

    # Transformer block - MultiHeadAttention + FFN + LayerNorm + Dropout
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=transformer_dim)(x, x)
    attention_output = Dropout(dropout_rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attention_output)

    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(transformer_dim)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    # Pooling over sequence dimension (here seq_len=1, so this is simple)
    x = GlobalAveragePooling1D()(out2)

    # Dense layers after transformer
    x = Dense(100, activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(50, activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(10, activation='elu')(x)

    # Output layer for steering angle regression
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=optimizer)
    return model