import pandas as pd
times = pd.read_csv(
    "results.txt",
    header=None,
    names=[
        "threads",
        "size",
        "how",
        "time"], sep=" ")
times = times[times["how"] != "CRITICAL"]
times = times.pivot(index="size", columns=["threads", "how"], values="time")
times.plot(figsize=(16, 9))
