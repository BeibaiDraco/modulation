# alignlab

A small framework for alignment experiments in linear recurrent population models.

## Quickstart

```bash
pip install -e .
python -m alignlab.cli run --config configs/default_smallpc1_custom.yaml
python -m alignlab.cli sweep --config configs/default_smallpc1_custom.yaml --ranges 0.02 0.04 0.08 0.12 0.16
```

Outputs (figures + JSON) are saved in `outputs/`.
