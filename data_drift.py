import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Reference dataset
reference_data = pd.DataFrame({
    "person": [40, 42, 38, 45, 41, 39, 43, 40, 44, 42],
    "car": [25, 24, 27, 23, 26, 25, 24, 26, 23, 25],
    "dog": [10, 11, 9, 12, 10, 9, 11, 10, 12, 9],
})

# Current / new data
current_data = pd.DataFrame({
    "person": [70, 72, 68, 75, 71, 69, 73, 70, 74, 72],
    "car": [10, 11, 9, 12, 10, 9, 11, 10, 12, 9],
    "dog": [5, 6, 4, 7, 5, 4, 6, 5, 7, 4],
})

# Create Data Drift report
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

# Run the drift analysis
snapshot = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save the report
snapshot.save_html("outputs/data_drift_report.html")

print("Data drift analysis completed successfully!")
print("Report saved to: outputs/data_drift_report.html")