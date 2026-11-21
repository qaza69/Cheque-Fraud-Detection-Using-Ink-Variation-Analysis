"""Run the three trained detectors over the demo cheques in data/samples/.

File names tell the expected answer: fraud_* must be flagged FRAUD, genuine_* must
pass as GENUINE.  The demo cheques come from IDRBT, which the final models were
trained on, so treat the numbers as a smoke test; the honest accuracy estimate is
the cross-validation result stored in outputs/models/ink_metrics.json.

    python test_detection.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.fraud_pipeline import FraudDetector   # noqa: E402

SAMPLE_DIR = os.path.join(HERE, "data", "samples")
MODEL_NAMES = ("classical", "siamese", "cnn")
IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".tif")


def expected_verdict(file_name):
    return "GENUINE" if file_name.startswith("genuine") else "FRAUD"


def main():
    images = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(IMAGE_TYPES))
    print(f"\nDemo-cheque check: {len(images)} images\n")

    scores = {}                                   # model -> (right, total)
    misses = []
    for name in MODEL_NAMES:
        try:
            detector = FraudDetector(model_type=name)
        except Exception as err:                  # e.g. model file not trained yet
            print(f"[skipped] {name}: {err}")
            continue

        print(f"{name} (threshold {detector.threshold:.3f})")
        right = 0
        for image in images:
            want = expected_verdict(image)
            try:
                result = detector.detect(os.path.join(SAMPLE_DIR, image), rgb=False)
                got, value = result["verdict"], result["score"]
            except Exception as err:
                got, value = f"ERROR: {err}", float("nan")
            hit = got == want
            right += hit
            if not hit:
                misses.append((name, image, want, got))
            print(f"  {'ok  ' if hit else 'MISS'}  {image:<34s} {got:<8s} score {value:.3f}")
        scores[name] = (right, len(images))
        print()

    if not scores:
        print("No model could be loaded - run `python -m src.train` first.")
        sys.exit(1)

    print("Summary")
    for name, (right, total) in scores.items():
        print(f"  {name:<10s} {right}/{total} correct ({100 * right / total:.0f}%)")
    for name, image, want, got in misses:
        print(f"  miss: [{name}] {image} expected {want}, got {got}")
    print("\nAll demo cheques classified correctly." if not misses else "")


if __name__ == "__main__":
    main()
