import time
import argparse
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    print(f"🚀 Starting FAKE training job for Fold {args.fold}...")

    # Simulate a 30-second training loop (5 epochs * 6 seconds)
    for epoch in range(5):
        print(f"Epoch {epoch}/5 - Crushing numbers...", flush=True)
        
        # Log a fake decreasing loss to MLflow
        fake_loss = 1.0 / (epoch + 1)
        mlflow.log_metric(f"fold_{args.fold}_fake_loss", fake_loss, step=epoch)
        
        time.sleep(6)

    print(f"✅ Finished FAKE training for Fold {args.fold}!")

if __name__ == "__main__":
    main()