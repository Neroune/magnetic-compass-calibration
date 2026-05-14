import math
import webbrowser
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
BAD_LIMIT = 0.20

SEED = 67
GENERATIONS = 700
POPULATION = 90
ELITE = 10


def project_path(file_name):
    return PROJECT_DIR / file_name


def read_points(file_name):
    return np.loadtxt(project_path(file_name), delimiter=",")


def lengths(points):
    return np.linalg.norm(points, axis=1)


def scale_to_mean_field(points):
    # Общий масштаб для всего набора. Не делим каждую точку на ее длину.
    return points / lengths(points).mean()


def correct(points, bias, scale):
    return (points - bias) / scale


def rms_error(points):
    err = lengths(points) - 1.0
    return float(np.sqrt(np.mean(err * err)))


def print_result(name, points):
    err = np.abs(lengths(points) - 1.0)
    bad = int(np.sum(err > BAD_LIMIT))
    print(f"{name:20s} RMS={rms_error(points):.6f}  mean={err.mean():.6f}  max={err.max():.6f}  bad={bad}")


def calibrate_mnk(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    h = np.column_stack([-2 * x, y * y, -2 * y, z * z, -2 * z, np.ones(len(points))])
    target = -(x * x)
    c1, c2, c3, c4, c5, c6 = np.linalg.lstsq(h, target, rcond=None)[0]

    bx = c1
    by = c3 / c2
    bz = c5 / c4
    alpha = bx * bx + c2 * by * by + c4 * bz * bz - c6

    sx = math.sqrt(alpha)
    sy = math.sqrt(alpha / c2)
    sz = math.sqrt(alpha / c4)

    bias = np.array([bx, by, bz])
    scale = np.array([sx, sy, sz])
    return correct(points, bias, scale), bias, scale


def unpack(theta):
    bias = theta[:3]
    scale = np.exp(theta[3:])
    return bias, scale


def score(points, theta):
    bias, scale = unpack(theta)
    calibrated = correct(points, bias, scale)
    penalty = 0.01 * float(np.mean(theta[3:] * theta[3:]))
    return rms_error(calibrated) + penalty


def clip(theta):
    theta[:3] = np.clip(theta[:3], -0.8, 0.8)
    theta[3:] = np.clip(theta[3:], math.log(0.4), math.log(2.0))
    return theta


def calibrate_generative(points, start_bias, start_scale):
    rng = np.random.default_rng(SEED)
    start = np.r_[start_bias, np.log(start_scale)]
    step = np.array([0.25, 0.25, 0.25, 0.18, 0.18, 0.18])

    population = [start]
    while len(population) < POPULATION:
        population.append(clip(start + rng.normal(0.0, step)))
    population = np.array(population)

    best = start.copy()
    best_score = score(points, best)

    for gen in range(GENERATIONS):
        scores = np.array([score(points, item) for item in population])
        order = np.argsort(scores)
        population = population[order]

        if scores[order[0]] < best_score:
            best = population[0].copy()
            best_score = float(scores[order[0]])

        mutation = 1.0 - 0.85 * gen / max(1, GENERATIONS - 1)
        children = [population[i].copy() for i in range(ELITE)]

        while len(children) < POPULATION:
            a = population[rng.integers(0, POPULATION // 3)]
            b = population[rng.integers(0, POPULATION // 3)]
            mix = rng.random()
            child = mix * a + (1.0 - mix) * b
            child += rng.normal(0.0, step * mutation)
            children.append(clip(child))

        population = np.array(children)

    bias, scale = unpack(best)
    return correct(points, bias, scale), bias, scale


def save_points(file_name, points):
    with project_path(file_name).open("w", encoding="utf-8") as f:
        for x, y, z in points:
            f.write(f"{x:.9f}, {y:.9f}, {z:.9f}\n")


def print_worst(raw, calibrated, title):
    err = lengths(calibrated) - 1.0
    indexes = np.argsort(np.abs(err))[::-1][:6]

    print(f"\nХудшие точки после {title}:")
    for i in indexes:
        x, y, z = raw[i]
        print(f"  строка {i + 1:3d}: ошибка={err[i]: .6f}, сырая=({x:.0f}, {y:.0f}, {z:.0f})")


def save_interactive_plot(file_name, title, points):
    import plotly.graph_objects as go

    err = np.abs(lengths(points) - 1.0)
    good = err <= BAD_LIMIT
    limit = max(1.1, float(np.max(np.abs(points))) * 1.05)

    u = np.linspace(0, 2 * math.pi, 50)
    v = np.linspace(0, math.pi, 25)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_surface(
        x=sphere_x,
        y=sphere_y,
        z=sphere_z,
        opacity=0.18,
        showscale=False,
        colorscale=[[0, "lightgray"], [1, "lightgray"]],
        name="единичная сфера",
        hoverinfo="skip",
    )
    fig.add_scatter3d(
        x=points[good, 0],
        y=points[good, 1],
        z=points[good, 2],
        mode="markers",
        marker={"size": 3, "color": "#1f77b4"},
        name="точки",
    )
    fig.add_scatter3d(
        x=points[~good, 0],
        y=points[~good, 1],
        z=points[~good, 2],
        mode="markers",
        marker={"size": 5, "color": "red"},
        name="ошибка > 0.20",
    )
    fig.update_layout(
        title=title,
        scene={
            "aspectmode": "cube",
            "xaxis": {"title": "X", "range": [-limit, limit]},
            "yaxis": {"title": "Y", "range": [-limit, limit]},
            "zaxis": {"title": "Z", "range": [-limit, limit]},
        },
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    fig.write_html(project_path(file_name), include_plotlyjs=True)


def open_files(file_names):
    for file_name in file_names:
        webbrowser.open_new(project_path(file_name).as_uri())


def main():
    input_file = "materials/12.txt"
    mnk_file = "12_mnk.txt"
    gen_file = "12_generative.txt"

    raw = read_points(input_file)
    data = scale_to_mean_field(raw)

    mnk, mnk_bias, mnk_scale = calibrate_mnk(data)
    gen, gen_bias, gen_scale = calibrate_generative(data, mnk_bias, mnk_scale)

    save_points(mnk_file, mnk)
    save_points(gen_file, gen)
    save_points("12_before_scaled.txt", data)
    save_interactive_plot("graph_before.html", "До корректировки", data)
    save_interactive_plot("graph_mnk.html", "После МНК", mnk)
    save_interactive_plot("graph_generative.html", "После генеративного алгоритма", gen)

    open_files(["graph_before.html", "graph_mnk.html", "graph_generative.html"])

    print(f"Файлы: {mnk_file}, {gen_file}")
    print("До калибровки: 12_before_scaled.txt")
    print("3D-окна: graph_before.html, graph_mnk.html, graph_generative.html\n")
    print_result("До", data)
    print_result("После МНК", mnk)
    print_result("После генеративного", gen)

    print("\nМНК bias/scale:")
    print(np.round(mnk_bias, 6), np.round(mnk_scale, 6))
    print("\nГенеративный bias/scale:")
    print(np.round(gen_bias, 6), np.round(gen_scale, 6))

    print_worst(raw, mnk, "МНК")
    print_worst(raw, gen, "генеративного")


if __name__ == "__main__":
    main()
