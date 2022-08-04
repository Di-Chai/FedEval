## BenchMarks

### Radar Chart

![Radar Plots](../images/radar.png)

**Privacy score (qualitative):**

- Score {math}`0` if directly exchanging raw data.
- Score {math}`1` if the exchanged parameters are in plaintext and calculated from a single round of training.
- Score {math}`2` if the exchanged parameters are in plaintext and calculated from multiple rounds of training.
- Score {math}`3` if the exchanged parameters are protected by DP or the parameters are compressed.
- Score {math}`4` if the exchanged parameters are protected by secure aggregation.
- Score {math}`5` if the exchanged parameters are protected by homomorphic encryption.

**Robustness score (quantitative):**

- Add {math}`3` points if the non-IID performance disparity to IID model {math}`\le 1\%`.
- Add {math}`2` points if the non-IID performance disparity to IID model {math}`\le 3\%`.
- Add {math}`1` points if the non-IID performance disparity to IID model {math}`\le 5\%`.
- Add {math}`1` point if the stragglers are handled.
- Add {math}`1` point if the dropouts are handled.

**Efficiency score (quantitative):** The efficiency score is the calculated through averaging the quantitative score of three sub-metrics, which are the communication rounds, communication amount, and the time consumption. For each of these sub-metrics, we choose one baseline model (i.e., FedSGD), score 1 point to the baseline model, and then compute the score of other methods through comparing to the baseline model. Specifically:

- If the method A's performance {math}`P_a` (e.g., time consumption) is worse than the baseline model {math}`P_b`, then we give score {math}`e^{1-P_b/P_a}` to A.
- Otherwise, if method A's performance {math}`P_a` is better than the baseline model {math}`P_b`, denoting the best performance as {math}`\bar{P}`, then we give score {math}`1 + 4(P_b - P_a)/(P_b - \bar{P})` to A. We set the best performance {math}`\bar{P}=0` for time consumption, {math}`\bar{P}=1` for communication rounds, and {math}`\bar{P}=0` for communication amount.

**Effectiveness score (quantitative):**

- If the model's performance is better than local model, we score according to the performance disparity to central model. Score {math}`5 \sim 1` if performance disparity to central model is {math}`\le 1\%`, {math}`\le 3\%`, {math}`\le 5\%`, {math}`\le 10\%`, and {math}`\le 20\%`, respectively.
- Score {math}`0` if the performance is worse or equal to local model.


### Parameter Tuning

TBD.

