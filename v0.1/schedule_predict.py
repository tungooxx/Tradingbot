"""
SCHEDULED PREDICTION RUNNER
===========================
Runs predict.py at 3:30 AM Vietnam time every day.

Setup Instructions:
1. Windows Task Scheduler:
   - Open Task Scheduler
   - Create Basic Task
   - Name: "Trading Predictions"
   - Trigger: Daily at 3:30 AM
   - Action: Start a program
   - Program: python
   - Arguments: schedule_predict.py
   - Start in: (your project directory)

2. Linux/Mac cron:
   - Run: crontab -e
   - Add: 30 3 * * * cd /path/to/project && python schedule_predict.py

3. Python schedule library (runs continuously):
   - Just run: python schedule_predict.py
   - Keeps running and executes at 3:30 AM daily
"""

import schedule
import time
import subprocess
import sys
import os
from datetime import datetime
import pytz

# Vietnam timezone
VIETNAM_TZ = pytz.timezone('Asia/Ho_Chi_Minh')


def run_predictions():
    """Run the prediction script"""
    vietnam_time = datetime.now(VIETNAM_TZ)
    print(f"\n{'=' * 60}")
    print(f"⏰ Scheduled run at {vietnam_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"{'=' * 60}\n")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    predict_script = os.path.join(script_dir, "predict.py")

    try:
        # Run predict.py
        result = subprocess.run(
            [sys.executable, predict_script],
            cwd=script_dir,
            capture_output=True,
            text=True
        )

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print(f"\n✅ Predictions completed successfully")
        else:
            print(f"\n❌ Predictions failed with return code {result.returncode}")

    except Exception as e:
        print(f"❌ Error running predictions: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main scheduling loop"""
    print("=" * 60)
    print("📅 PREDICTION SCHEDULER")
    print("=" * 60)
    print("This script will run predictions at 3:30 AM Vietnam time daily")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Schedule the job
    schedule.every().day.at("03:30").do(run_predictions)

    # Also run immediately if it's close to 3:30 AM (for testing)
    current_time = datetime.now(VIETNAM_TZ)
    if current_time.hour == 3 and current_time.minute >= 25:
        print("\n⏰ It's close to 3:30 AM, running predictions now...")
        run_predictions()

    # Keep running
    print(f"\n⏰ Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("⏳ Waiting for next scheduled run at 3:30 AM Vietnam time...")
    print("   (You can also run predict.py manually anytime)\n")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Scheduler stopped by user")
        sys.exit(0)

