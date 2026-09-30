# First-Day Checklist

- Verify CSPB.ML.2018R2 files and truth file
- Verify .tim binary dtype from official reader/documentation
- Read 10–20 .tim files
- Confirm I/Q values look valid
- Match each file to truth metadata
- Plot I, Q, magnitude and PSD
- Build file-level 70/15/15 split
- Save train/val/test manifests
- Confirm zero signal overlap across splits
- Produce [2,2048] windows
- Run one batch through ResNet
- Confirm output [batch,8]
- Overfit a tiny subset
- Then start full training
