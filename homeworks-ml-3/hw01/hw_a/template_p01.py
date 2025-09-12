import numpy as np


def softmax(vector):
    """
    vector: np.array of shape (n, m)

    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    """
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_


def multiplicative_attention_score(decoder_hidden_state, encoder_hidden_states, W_mult):
    encoder_hidden_vector = np.dot(W_mult, encoder_hidden_states)
    attention_scores = np.dot(decoder_hidden_state.T, encoder_hidden_vector)
    return attention_scores


def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    # your code here
    softmax_vector = softmax(
        multiplicative_attention_score(
            decoder_hidden_state, encoder_hidden_states, W_mult
        )
    )
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T
    return attention_vector


def additive_attention_score(
    decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec
):
    encoder_proj = np.dot(W_add_enc, encoder_hidden_states)
    decoder_proj = np.dot(W_add_dec, decoder_hidden_state)
    sum_proj = encoder_proj + decoder_proj
    tanh_res = np.tanh(sum_proj)
    attention_scores = np.dot(v_add.T, tanh_res)
    return attention_scores


def additive_attention(
    decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec
):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    # your code here
    softmax_vector = softmax(
        additive_attention_score(
            decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec
        )
    )
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T
    return attention_vector
