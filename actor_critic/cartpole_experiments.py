import keras
import matplotlib
import numpy as np
import plotly.graph_objects as go

matplotlib.use("Agg")

if (backend := keras.backend.backend()) != "torch":
    raise NotImplementedError(f"Backend {backend!r} nie jest wspierany!")
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[backend] PyTorch  |  urzadzenie: {DEVICE.upper()}")


MODELS = [
    "results/cartpole_default",
    "results/cartpole_dual_head",
    "results/cartpole_large_lr",
    "results/cartpole_small_net",
]


def eval_model(model, X, V, A, W):
    # Box([-4.8 -inf -0.41887903 -inf], [4.8 inf 0.41887903 inf], (4,), float32)
    # 0: x position of the cart -4.8 to 4.8
    # 1: x velocity of the cart -inf to inf
    # 2: angle of the pole -0.41887903 to 0.418 radians (-24 to 24 degrees)
    # 3: angular velocity of the pole -inf to inf

    states = np.stack(
        [
            X,  # x
            V,  # velocity
            A,  # angle
            W,  # angular_velocity
        ],
        axis=-1,
    )

    state_tensor = torch.tensor(
        states.reshape(-1, 4), dtype=torch.float32, device=DEVICE
    )

    with torch.no_grad():
        logits, values = model(state_tensor)

    return values.detach().cpu().numpy()


def plotly_3d_plot(xs, ys, zs, title, x_label, y_label, z_label, save_path):
    fig = go.Figure(
        data=[
            go.Surface(
                x=xs,
                y=ys,
                z=zs,
                colorscale="Viridis",
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
    )

    fig.write_html(save_path)


def experiment(model_path: str):
    model = keras.saving.load_model(f"{model_path}/cartpole_final.keras")
    model.summary()

    positions = np.linspace(-4.8, 4.8, 100)
    velocities = np.linspace(-5.0, 5.0, 100)

    X, V = np.meshgrid(positions, velocities, indexing="xy")

    values = eval_model(model, X, V, np.zeros_like(X), np.zeros_like(V)).reshape(
        100, 100
    )

    plotly_3d_plot(
        xs=X,
        ys=V,
        zs=values,
        title=f"Wartości stanu dla modelu {model_path}",
        x_label="Pozycja wózka",
        y_label="Prędkość wózka",
        z_label="Wartość stanu",
        save_path=f"{model_path}/value_function_positions.html",
    )

    angles = np.linspace(-0.41887903, 0.41887903, 100)
    angular_velocities = np.linspace(-5.0, 5.0, 100)

    A, W = np.meshgrid(angles, angular_velocities, indexing="xy")

    values = eval_model(model, np.zeros_like(A), np.zeros_like(W), A, W).reshape(
        100, 100
    )

    plotly_3d_plot(
        xs=A,
        ys=W,
        zs=values,
        title=f"Wartości stanu dla modelu {model_path}",
        x_label="Kąt drążka",
        y_label="Prędkość kątowa drążka",
        z_label="Wartość stanu",
        save_path=f"{model_path}/value_function_angle.html",
    )


if __name__ == "__main__":
    for model_path in MODELS:
        print(f"Przeprowadzanie eksperymentu dla modelu: {model_path}")
        experiment(model_path)
