"""Train a late-delivery model and build a daily shipment review queue."""

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from random import Random

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_FIELDS = ("sla_hours_remaining", "hub_dwell_hours", "lane_backlog", "distance_km")
INPUT_FIELDS = ("shipment_id", "snapshot_date", *NUMERIC_FIELDS, "service_level")
SERVICE_LEVELS = ("standard", "express")
START_DATE = date(2025, 1, 1)
TRAIN_DAYS = 60
TEST_DAYS = 20
SHIPMENTS_PER_DAY = 40
DEMO_CAPACITY = 10


@dataclass(frozen=True)
class Shipment:
    shipment_id: str
    snapshot_date: date
    sla_hours_remaining: float
    hub_dwell_hours: float
    lane_backlog: float
    distance_km: float
    service_level: str

    def features(self):
        return {
            **{name: getattr(self, name) for name in NUMERIC_FIELDS},
            "service_level": self.service_level,
        }


def synthetic_shipments(start, days, seed):
    """Return in-transit snapshots and simulated outcomes as separate lists."""
    random = Random(seed)
    shipments, late = [], []
    for day in range(days):
        snapshot_date = start + timedelta(days=day)
        network_pressure = random.uniform(0, 16)
        for number in range(1, SHIPMENTS_PER_DAY + 1):
            shipment = Shipment(
                shipment_id=f"S-{snapshot_date:%Y%m%d}-{number:02d}",
                snapshot_date=snapshot_date,
                sla_hours_remaining=round(random.uniform(5, 45), 1),
                hub_dwell_hours=round(min(random.expovariate(1 / 6), 27), 1),
                lane_backlog=round(random.uniform(0, 65) + network_pressure),
                distance_km=random.randint(60, 1200),
                service_level=random.choices(SERVICE_LEVELS, weights=(4, 1))[0],
            )
            # Simulate remaining transit time with noise and occasional disruptions.
            remaining_hours = (
                6 + shipment.distance_km / 100
                + 0.55 * shipment.hub_dwell_hours + 0.12 * shipment.lane_backlog
                - 4 * (shipment.service_level == "express") + random.gauss(0, 7)
            )
            if random.random() < 0.08:
                remaining_hours += random.uniform(3, 15)
            shipments.append(shipment)
            late.append(int(remaining_hours > shipment.sla_hours_remaining))
    return shipments, late


def train_model():
    shipments, late = synthetic_shipments(START_DATE, TRAIN_DAYS, seed=17)
    model = make_pipeline(DictVectorizer(sparse=False), StandardScaler(), LogisticRegression())
    model.fit([shipment.features() for shipment in shipments], late)
    return model


def predict_risk(model, shipments):
    return model.predict_proba([shipment.features() for shipment in shipments])[:, 1]


def review_queue(shipments, scores, capacity):
    by_date = defaultdict(list)
    for shipment, score in zip(shipments, scores, strict=True):
        by_date[shipment.snapshot_date].append((shipment, score))

    for snapshot_date in sorted(by_date):
        daily_queue = sorted(by_date[snapshot_date], key=lambda item: (-item[1], item[0].shipment_id))
        for rank, (shipment, score) in enumerate(daily_queue[:capacity], start=1):
            yield rank, shipment, score


def evaluate(shipments, late, risk, capacity):
    outcomes = {shipment: label for shipment, label in zip(shipments, late, strict=True)}
    total_late = sum(late)
    print(f"Evaluation: {len(shipments)} shipments, {total_late} late, {capacity} reviews per day")
    print(f"{'Approach':<20} {'Late caught':>12} {'Precision':>10} {'Recall':>8} {'Avg precision':>14}")
    for name, scores in (
        ("Logistic regression", risk),
        ("Least SLA time first", [-shipment.sla_hours_remaining for shipment in shipments]),
    ):
        queue = list(review_queue(shipments, scores, capacity))
        caught = sum(outcomes[shipment] for _, shipment, _ in queue)
        print(f"{name:<20} {caught:>6}/{len(queue):<5} {caught / len(queue):>10.1%} "
              f"{caught / total_late:>8.1%} {average_precision_score(late, scores):>14.3f}")


def follow_up(shipment):
    """Map observed conditions to checks a dispatcher can perform."""
    if shipment.sla_hours_remaining <= 8:
        return "8 hours or less until SLA", "Confirm an updated ETA with the carrier"
    if shipment.hub_dwell_hours >= 12:
        return "Hub dwell of at least 12 hours", "Check release status with the hub"
    if shipment.lane_backlog >= 55:
        return "Lane backlog of at least 55 shipments", "Confirm the next lane departure"
    return "Review in model rank order", "Verify the next tracking milestone"


def write_queue(path, shipments, scores, capacity):
    fields = (*INPUT_FIELDS, "daily_rank", "late_risk", "observed_signal", "suggested_action")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, shipment, risk in review_queue(shipments, scores, capacity):
            signal, action = follow_up(shipment)
            writer.writerow({
                **{name: getattr(shipment, name) for name in INPUT_FIELDS},
                "daily_rank": rank,
                "late_risk": f"{risk:.4f}",
                "observed_signal": signal,
                "suggested_action": action,
            })


def read_shipments(path):
    shipments, seen_snapshots = [], set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(INPUT_FIELDS).issubset(columns):
            raise ValueError(f"CSV needs unique columns including {', '.join(INPUT_FIELDS)}")
        for row in reader:
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("row does not match the CSV header")
                shipment_id = row["shipment_id"].strip()
                if not shipment_id:
                    raise ValueError("shipment_id must be nonempty")
                snapshot_date = date.fromisoformat(row["snapshot_date"])
                if snapshot_date.isoformat() != row["snapshot_date"]:
                    raise ValueError("snapshot_date must use YYYY-MM-DD")
                snapshot_key = (snapshot_date, shipment_id)
                if snapshot_key in seen_snapshots:
                    raise ValueError("shipment_id must be unique per snapshot_date")
                service_level = row["service_level"].strip().lower()
                if service_level not in SERVICE_LEVELS:
                    raise ValueError("service_level must be standard or express")
                values = {name: float(row[name]) for name in NUMERIC_FIELDS}
                for name, value in values.items():
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"{name} must be a finite, nonnegative number")
                shipments.append(Shipment(shipment_id, snapshot_date, **values, service_level=service_level))
                seen_snapshots.add(snapshot_key)
            except ValueError as error:
                raise ValueError(f"CSV line {reader.line_num}: {error}") from error
    if not shipments:
        raise ValueError("CSV contains no shipments")
    return shipments


def positive_integer(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("capacity must be at least 1")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Evaluate on held-out days and write a sample queue")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo_queue.csv"))
    score = commands.add_parser("score", help="Rank an unlabeled CSV of in-transit shipments")
    score.add_argument("--input", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    score.add_argument("--capacity", type=positive_integer, default=DEMO_CAPACITY)
    args = parser.parse_args()

    try:
        if args.command == "demo":
            model = train_model()
            test_start = START_DATE + timedelta(days=TRAIN_DAYS)
            held_out, late = synthetic_shipments(test_start, TEST_DAYS, seed=29)
            evaluate(held_out, late, predict_risk(model, held_out), DEMO_CAPACITY)
            demo_start = test_start + timedelta(days=TEST_DAYS)
            shipments, _ = synthetic_shipments(demo_start, days=1, seed=43)
            capacity = DEMO_CAPACITY
        else:
            if args.input.resolve() == args.output.resolve():
                raise ValueError("choose distinct paths for input and output")
            shipments = read_shipments(args.input)
            model = train_model()
            capacity = args.capacity
        write_queue(args.output, shipments, predict_risk(model, shipments), capacity)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    print(f"Wrote review queue to {args.output}")


if __name__ == "__main__":
    main()
