Post 1
Automated scoring in prompt injection benchmarks has a dirty secret: nobody checks the false positive rate. We human-labeled 324 injection trials and found 7 phantom successes the scorer invented. One model looked 4x more vulnerable than it actually was.

Post 2
The setup: 324 trials, 3 models, 12 injection payloads, automated 0-3 severity scorer. Scorer flagged 37 as injection successes. Human review confirmed 30. Precision: 0.811. After tuning: 1.000 precision, 1.000 recall. Cost of calibration: 30 minutes.

Post 3
The wildest false positives: 3 rows scored as "full compromise" (score 3) where the model was actually *defending* the user. It detected the injection, alerted a trusted admin, ran diagnostics. The scorer saw "tool call + injection payload" and assumed the worst.

Post 4
Impact on model rankings: Kimi K2.5 went from 22% raw injection rate (automated) to 5.6% (human-labeled). That's the difference between "about as vulnerable as Haiku" and "most resistant model tested." Seven false positives, six from one model, changed the leaderboard.

Post 5
If you publish injection benchmark numbers, report your scorer's precision and recall against human labels. It's an hour of work and it might save you from publishing wrong conclusions. Data + calibration scripts: github.com/misty-step/laboratory
