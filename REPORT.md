# REPORT — Lakeshore Bike Share support lead

Draft structure only. Replace every bracketed field with your measured results and your own interpretation. Final report should be 300–450 words. Do not submit this template unchanged.

## Result
At 300 training examples, the strongest baseline achieved [mean validation accuracy and range] and [test accuracy and macro F1], compared with [best neural variant, validation mean/range and test scores]. [Explain which method won and by how much.]

## Learning curve
At 50, 100 and 300 rows, [compare means and ranges for the four methods using reports/summary.csv and learning_curve.png]. [Explain where fine-tuning helped or did not, and whether ranges overlap.]

## Reasons and checks
[Discuss at least two possible reasons. Check sample-size effects using the learning curve, seed variability using min/max, and learning rate/epoch effects using tuning.csv and epoch logs. Distinguish measured checks from hypotheses about repeated patterns and vague or mixed-topic data.]

## Independent messages
[Compare accuracy and macro F1 on your 40 original messages and held-out generated test tickets, using final.json. Discuss whether the difference suggests synthetic templates flatter the models, and note that 40 messages are a small sample.]

## Recommendation and limitation
[Recommend the method justified by evidence, considering cost, and state what new evidence would change the recommendation. Explain why invented data and repeated templates cannot establish accuracy on real customer tickets.]
