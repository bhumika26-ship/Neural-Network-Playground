import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# Page settings
st.set_page_config(
    page_title="Neural Network Playground",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 Neural Network Playground")
st.write("Understand how a neural network learns the XOR problem.")  

st.divider()

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

st.subheader("1. XOR Dataset")

st.write(
    "XOR gives output 1 when the inputs are different, "
    "and 0 when they are the same."
)

st.dataframe({
    "Input A": X[:, 0],
    "Input B": X[:, 1],
    "Expected Output": y
})

# Model selection
st.subheader("2. Choose Your Neural Network")

model_name = st.selectbox(
    "Select Model",
    ["Single Perceptron", "Neural Network (MLP)"]
)

neurons = st.selectbox(
    "Hidden Layer Neurons",
    [1, 2, 4, 8],
    index=2
)

# Train selected model
if model_name == "Single Perceptron":
    model = Perceptron(
        max_iter=2000,
        random_state=42
    )
else:
    model = MLPClassifier(
        hidden_layer_sizes=(neurons,),
        activation="tanh",
        solver="lbfgs",
        max_iter=5000,
        random_state=42
    )

model.fit(X, y)

# Predictions
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)

st.subheader("3. Model Performance")

col1, col2 = st.columns(2)

col1.metric(
    "Training Accuracy",
    f"{accuracy * 100:.1f}%"
)

col2.metric(
    "Correct Predictions",
    f"{int(np.sum(predictions == y))} / {len(y)}"
)

st.dataframe({
    "Input A": X[:, 0],
    "Input B": X[:, 1],
    "Expected Output": y,
    "Predicted Output": predictions
})

# Decision boundary
st.subheader("4. Decision Boundary Visualization")

fig, ax = plt.subplots(figsize=(7, 5))

DecisionBoundaryDisplay.from_estimator(
    model,
    X,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    alpha=0.35,
    ax=ax
)

ax.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    edgecolors="black",
    s=180
)

ax.set_xlabel("Input A")
ax.set_ylabel("Input B")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_title(model_name)

st.pyplot(fig)
plt.close(fig)

# Explanation
st.subheader("5. What Did You Learn?")

if model_name == "Single Perceptron":
    st.warning(
        "A single perceptron cannot correctly classify "
        "all XOR inputs because XOR is not linearly separable."
    )
else:
    st.info(
        "An MLP uses a hidden layer to learn nonlinear "
        "patterns. Check the predictions to see whether "
        "the model learned XOR correctly."
    )

st.caption(
    "Built with Python, NumPy, Matplotlib, "
    "scikit-learn and Streamlit."
)
