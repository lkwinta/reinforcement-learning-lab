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
    "results/lunar_default",
]


def eval_model(model, X, Y, V_X, V_Y, A, W, L, R):
    # Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ], [ 2.5 2.5 10. 10. 6.2831855 10. 1. 1. ], (8,), float32)
    # 0: x position of the lander -2.5 to 2.5
    # 1: y position of the lander -2.5 to 2.5
    # 2: x velocity of the lander -10 to 10
    # 3: y velocity of the lander -10 to 10
    # 4: angle of the lander -6.2831855 to 6.2831855 radians (-360 to 360 degrees)
    # 5: angular velocity of the lander -10 to 10
    # 6: left leg contact with the ground
    # 7: right leg contact with the ground

    states = np.stack(
        [
            X,  # x
            Y,  # y
            V_X,  # x velocity
            V_Y,  # y velocity
            A,  # angle
            W,  # angular_velocity
            L,  # left leg contact
            R,  # right leg contact
        ],
        axis=-1,
    )

    state_tensor = torch.tensor(
        states.reshape(-1, 8), dtype=torch.float32, device=DEVICE
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
    model = keras.saving.load_model(f"{model_path}/lunar_final.keras")
    model.summary()

    positions_x = np.linspace(-2.5, 2.5, 100)
    positions_y = np.linspace(-2.5, 2.5, 100)

    X, Y = np.meshgrid(positions_x, positions_y, indexing="xy")

    values = eval_model(
        model,
        X,
        Y,
        np.zeros_like(X),
        np.zeros_like(Y),
        np.zeros_like(X),
        np.zeros_like(Y),
        np.zeros_like(X),
        np.zeros_like(Y),
    ).reshape(100, 100)

    plotly_3d_plot(
        xs=X,
        ys=Y,
        zs=values,
        title=f"Wartości stanu dla modelu {model_path}",
        x_label="Pozycja X lądownika",
        y_label="Pozycja Y lądownika",
        z_label="Wartość stanu",
        save_path=f"{model_path}/value_function_positions.html",
    )

    angles = np.linspace(-6.2831855, 6.2831855, 100)
    angular_velocities = np.linspace(-10.0, 10.0, 100)

    A, W = np.meshgrid(angles, angular_velocities, indexing="xy")

    values = eval_model(
        model,
        np.zeros_like(A),
        np.zeros_like(W),
        A,
        W,
        np.zeros_like(A),
        np.zeros_like(W),
        np.zeros_like(A),
        np.zeros_like(W),
    ).reshape(100, 100)

    plotly_3d_plot(
        xs=A,
        ys=W,
        zs=values,
        title=f"Wartości stanu dla modelu {model_path}",
        x_label="Kąt lądownika",
        y_label="Prędkość kątowa lądownika",
        z_label="Wartość stanu",
        save_path=f"{model_path}/value_function_angle.html",
    )


if __name__ == "__main__":
    for model_path in MODELS:
        print(f"Przeprowadzanie eksperymentu dla modelu: {model_path}")
        experiment(model_path)
