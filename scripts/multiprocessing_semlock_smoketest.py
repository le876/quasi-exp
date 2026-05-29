from __future__ import annotations

import multiprocessing as mp


def _f(x: int) -> int:
    return x * x


def main() -> None:
    try:
        with mp.Pool(processes=2) as pool:
            out = pool.map(_f, [1, 2, 3, 4])
        print("OK multiprocessing.Pool:", out)
    except Exception as e:  # noqa: BLE001
        print("FAILED multiprocessing.Pool:", repr(e))


if __name__ == "__main__":
    main()
