from auto_yolo import envs

readme = "running yolo baseline"

durations = dict(
    long=dict(
        max_hosts=1, ppn=6, cpp=2, gpu_set="0,1", wall_time="24hours",
        project="rpp-bengioy", cleanup_time="20mins",
        slack_time="5mins", n_repeats=1, step_time_limit="24hours"),

    build=dict(
        max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="2hours",
        project="rpp-bengioy", cleanup_time="2mins",
        slack_time="2mins", n_repeats=1, step_time_limit="2hours",
        config=dict(do_train=False)),

    short=dict(
        max_hosts=1, ppn=2, cpp=2, gpu_set="0", wall_time="20mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, n_param_settings=4),

    small_oak=dict(
        max_hosts=1, ppn=4, cpp=2, gpu_set="0", wall_time="30mins",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":"),

    build_oak=dict(
        max_hosts=1, ppn=3, cpp=2, gpu_set="0", wall_time="1year",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":",
        config=dict(do_train=False)),

    oak=dict(
        max_hosts=1, ppn=1, cpp=2, gpu_set="0", wall_time="1year",
        project="rpp-bengioy", cleanup_time="1mins",
        slack_time="1mins", n_repeats=1, kind="parallel", host_pool=":",
        step_time_limit="1year"),
)

envs.run_experiment(
    "yolo_baseline_transfer", dict(), readme,
    alg="yolo_baseline_transfer", task="scatter", durations=durations,
)
