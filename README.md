# Labeling

This project implements the triple barrier events method from
"Advances in Financial Machine Learning" by Marcos Lopez de Prado.
It uses Python's `multiprocessing` library for parallel computations.

## Setup

1. Install Python 3.12 or newer.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -e .
   ```

## Project Structure

```
labeling/
  data/                 Sample data (HDF5 bars)
  utils/                Core labeling utilities
    cusum.py            CUSUM filter
    labeling.py         Triple barrier labeling logic
    multiprocessing.py  Parallel helpers
  label_bars.ipynb       Example notebook
```

## Functions in `utils/labeling.py`

- `apply_pt_sl_on_t1`: finds the first profit-taking or stop-loss touch time for each event in a slice of the index.
- `add_vertical_barrier`: builds vertical barrier timestamps using a fixed time offset from each event.
- `get_events`: creates the triple-barrier events table and applies PT/SL logic in parallel, with optional side.
- `get_bins`: computes event returns and labels (including meta-labeling) from triple-barrier events.
- `get_barriers_hit`: assigns labels based on which barrier was hit first (top, bottom, or vertical).
- `get_daily_volatility`: estimates daily volatility using EWM standard deviation of daily returns.
- `get_vertical_barriers`: returns vertical barrier timestamps `num_days` after each event.
- `get_barrier_events`: end-to-end pipeline using CUSUM events, volatility, and triple-barrier labeling.

## References

- Marcos Lopez de Prado, "Advances in Financial Machine Learning", Wiley, 2018.