# Shipment Risk Triage

A Python project that helps a logistics dispatcher decide which in-transit shipments to review first. A logistic regression model uses a daily shipment snapshot to estimate the probability of a late delivery. The ranked queue pairs that estimate with a suggested follow-up, such as checking a shipment's status with the hub.

## Run the demo

Requires Python 3.10 or newer.

```sh
python -m pip install -r requirements.txt
python triage.py demo
```

The demo generates reproducible synthetic shipment history, prints the holdout evaluation, and writes a ranked queue to `outputs/demo_queue.csv`.

## Score a shipment file

```sh
python triage.py score --input data/example_shipments.csv --output outputs/example_queue.csv --capacity 3
```

The scoring command trains on generated history each time and ranks up to three shipments per snapshot date in this example. Use `--capacity` to match the number of shipments your team can review each day. Provide one row per shipment per day, with fields available before the delivery deadline. Numeric values must be nonnegative.

In the example, `EX-102` ranks first with an estimated late-delivery risk of 96.3%. The queue flags that it has spent at least 12 hours at the hub and suggests checking its release status.

| CSV field | Meaning |
| --- | --- |
| `shipment_id` | Shipment identifier |
| `snapshot_date` | Review date in `YYYY-MM-DD` format |
| `sla_hours_remaining` | Hours remaining until the promised delivery time |
| `hub_dwell_hours` | Hours already spent at the hub |
| `lane_backlog` | Number of shipments waiting on the same lane |
| `distance_km` | Remaining distance in kilometers |
| `service_level` | `standard` or `express` |

## How the model is evaluated

The model trains on the first 60 synthetic days and is tested on the next 20. The comparison uses a dispatch rule that reviews the shipments with the least time remaining before their deadline. Both approaches get 10 reviews per day. Precision measures the share of selected shipments that arrive late, recall measures the share of all late shipments selected, and average precision summarizes ranking quality across thresholds.

On the 800 held-out shipments, 323 arrived late in the simulation. Each approach reviewed 200 shipments.

| Approach | Late shipments caught | Precision | Recall | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 184 | 92.0% | 57.0% | 0.909 |
| Least SLA time first | 171 | 85.5% | 52.9% | 0.853 |

## Using this with an operations team

The model supplies the risk estimate. Rules based on the shipment snapshot supply the follow-up suggestions, and a dispatcher decides what action to take. Before using the queue in production, backtest it on timestamped shipment records, check whether it improves on the team's current triage rule, and pilot it with dispatchers at their actual review capacity. The synthetic results show how the workflow can be evaluated.
