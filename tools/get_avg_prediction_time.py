import glob
import json
import argparse

import numpy as np

parser = argparse.ArgumentParser(description='Extract the average prediction\
                                    time with the specified trace directory')
parser.add_argument('trace_dir', type=str, help='Specify the trace directory')
args = parser.parse_args()

trace_dir = args.trace_dir
trace_path = trace_dir + "/*.ctr"


def parse_tracefile(filename):

    with open(filename, 'r') as f:
        raw = json.load(f)

    if 'traceEvents' not in raw:
        return None

    traceEvents = raw['traceEvents']

    timestamps = (
        (
            event['ts'],
            event['ts'] + event['dur']
        )
        for event in traceEvents
        if 'ts' in event and 'dur' in event
    )
    timestamps = sorted(timestamps, key=lambda x: x[1])

    min_ts = timestamps[0]
    max_ts = timestamps[-1]
    return max_ts[1] - min_ts[0]


durations = []

for filename in glob.glob(trace_path):
    # Exclude init0.ctr and prediction0.ctr as they don't reflect the real prediction time.
    if ("init0.ctr" not in filename) & ("prediction0.ctr" not in filename):
        duration = parse_tracefile(filename)
        durations.append(duration)

# Export duration for 100 predictions:
np.save(trace_dir + "prediction_durations_list.npy", durations)
print("Summary statistics for {} predictions:".format(len(durations)))
print('Average Prediction Time:', float(sum(durations)) / len(durations) / 1000, "ms")
print('Median Prediction Time:', np.median(durations) / 1000, "ms")
print('Standard Deviation Time:', np.std(durations) / 1000, "ms")
print('Min Prediction Time:', np.min(durations) / 1000, "ms")
print('Max Prediction Time:', np.max(durations) / 1000, "ms")
