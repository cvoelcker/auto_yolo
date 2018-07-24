from auto_yolo import envs
from auto_yolo.models import yolo_air

readme = "Running baseline for transfer experiment."

distributions = [
    dict(n_digits=1, cc_threshold=0.01),
    dict(n_digits=2, cc_threshold=0.01),
    dict(n_digits=3, cc_threshold=0.01),
    dict(n_digits=4, cc_threshold=0.01),
    dict(n_digits=5, cc_threshold=0.01),
    dict(n_digits=6, cc_threshold=0.01),
    dict(n_digits=7, cc_threshold=0.01),
    dict(n_digits=8, cc_threshold=0.01),
    dict(n_digits=9, cc_threshold=0.01),
    dict(n_digits=10, cc_threshold=0.01),
    dict(n_digits=11, cc_threshold=0.01),
    dict(n_digits=12, cc_threshold=0.01),
    dict(n_digits=13, cc_threshold=0.01),
    dict(n_digits=14, cc_threshold=0.01),
    dict(n_digits=15, cc_threshold=0.02),
    dict(n_digits=16, cc_threshold=0.17),
    dict(n_digits=17, cc_threshold=0.17),
    dict(n_digits=18, cc_threshold=0.18),
    dict(n_digits=19, cc_threshold=0.47),
    dict(n_digits=20, cc_threshold=0.47),
]

# count error
distributions = [
    dict(n_digits=1, cc_threshold=0.32),
    dict(n_digits=2, cc_threshold=0.36),
    dict(n_digits=3, cc_threshold=0.4),
    dict(n_digits=4, cc_threshold=0.37),
    dict(n_digits=5, cc_threshold=0.4299),
    dict(n_digits=6, cc_threshold=0.37),
    dict(n_digits=7, cc_threshold=0.63),
    dict(n_digits=8, cc_threshold=0.5),
    dict(n_digits=9, cc_threshold=0.71),
    dict(n_digits=10, cc_threshold=0.8),
    dict(n_digits=11, cc_threshold=0.88),
    dict(n_digits=12, cc_threshold=0.92),
    dict(n_digits=13, cc_threshold=0.87),
    dict(n_digits=14, cc_threshold=0.96),
    dict(n_digits=15, cc_threshold=0.85),
    dict(n_digits=16, cc_threshold=0.98),
    dict(n_digits=17, cc_threshold=0.98),
    dict(n_digits=18, cc_threshold=0.96),
    dict(n_digits=19, cc_threshold=0.47), # Redo
    dict(n_digits=20, cc_threshold=0.47), # Redo
]

# count 1norm
distributions = [
    dict(n_digits=1, cc_threshold=0.32), # Redo vvv
    dict(n_digits=2, cc_threshold=0.36),
    dict(n_digits=3, cc_threshold=0.4),
    dict(n_digits=4, cc_threshold=0.37),
    dict(n_digits=5, cc_threshold=0.4299),
    dict(n_digits=6, cc_threshold=0.37), # Redo ^^^
    dict(n_digits=7, cc_threshold=0.6),
    dict(n_digits=8, cc_threshold=0.57),
    dict(n_digits=9, cc_threshold=0.71),
    dict(n_digits=10, cc_threshold=0.74),
    dict(n_digits=11, cc_threshold=0.74),
    dict(n_digits=12, cc_threshold=0.82),
    dict(n_digits=13, cc_threshold=0.87),
    dict(n_digits=14, cc_threshold=0.88),
    dict(n_digits=15, cc_threshold=0.89),
    dict(n_digits=16, cc_threshold=0.91),
    dict(n_digits=17, cc_threshold=0.96),
    dict(n_digits=18, cc_threshold=0.98),
    dict(n_digits=19, cc_threshold=1.0),
    dict(n_digits=20, cc_threshold=1.0),
]

for d in distributions:
    n_digits = d['n_digits']
    d.update(
        min_chars=n_digits,
        max_chars=n_digits
    )


def build_net(scope):
    from dps.utils.tf import MLP
    return MLP([10, 10], scope=scope)


durations = dict(
    oak=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="1year",
        cleanup_time="1mins", slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":"),
)

config = dict(
    curriculum=[dict()],
    n_train=32, n_val=1000, stopping_criteria="AP,max", threshold=0.99,
    min_digits=1, max_digits=1, do_train=False,
    render_hook=yolo_air.YoloAir_ComparisonRenderHook(show_zero_boxes=False),
    build_object_encoder=build_net, build_object_decoder=build_net
)

envs.run_experiment(
    "transfer_baseline", config, readme, distributions=distributions,
    alg="baseline", task="scatter", durations=durations,
)