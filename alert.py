import os
from datetime import datetime
from database import is_flagged, log_plate

def check_and_alert(plate_number, vehicle_type, track_id, frame=None):
    if not plate_number:
        return False

    # check if plate is in watchlist
    reason = is_flagged(plate_number)
    flagged = reason is not None

    # log every plate to database
    log_plate(plate_number, vehicle_type, track_id, flagged)

    # if flagged trigger alert
    if flagged:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = f"🚨 ALERT [{timestamp}] Flagged plate detected: {plate_number} | Reason: {reason} | Vehicle: {vehicle_type} #{track_id}"
        print(alert_msg)

        # save alert to log file
        with open("output/alerts.txt", "a") as f:
            f.write(alert_msg + "\n")

        return True

    return False
