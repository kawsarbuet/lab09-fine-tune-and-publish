# Lab 09 Report — Lakeshore Bike Share

## 1. Result

At 300 training examples, TF-IDF with logistic regression, the strongest baseline, achieved 86.53% mean validation accuracy and 83.75% held-out test accuracy, while full DistilBERT fine-tuning achieved 90.42% and 88.33%, respectively, with test macro F1 scores of 0.8370 and 0.8857.

## 2. Where fine-tuning helped and where it did not

The [learning curve](reports/learning_curve.png) shows that TF-IDF outperformed full fine-tuning at 50 examples (67.92% versus 54.03%) and 100 examples (79.31% versus 78.06%). At 300 examples, full fine-tuning led by 3.89 percentage points. Its three-seed validation range (89.17–91.67%) did not overlap with TF-IDF's range (86.25–86.67%), although the ranges overlapped at 100 examples. Head-only training never surpassed TF-IDF, reaching 71.25% at 300 examples. The majority baseline remained at 28.75% throughout.

## 3. Possible reasons and checks

Limited training data likely made full fine-tuning less effective initially. Comparing the three sample sizes supports this explanation: its mean accuracy increased from 54.03% to 90.42%, while its seed-to-seed range narrowed. Repeated wording in the synthetic tickets may also favor TF-IDF, although a template-separated experiment was not conducted to isolate this effect.

The [tuning results](reports/tuning.csv) show that full fine-tuning at 100 examples improved from 47.92% with a learning rate of 0.00002 and two epochs to 72.08% with 0.00005 and three epochs. This suggests training settings matter, but changing both parameters together prevents isolating their individual effects.

## 4. Independent messages

On the 40 separate messages, full fine-tuning correctly classified 37 (92.50%; macro F1 0.9233), compared with TF-IDF's 35 (87.50%; macro F1 0.8736). Both exceeded their generated-test accuracies, so this comparison does not demonstrate that synthetic data flatters the models. The independent set is small, balanced and may contain clearer requests than real support traffic. Personal authorship of these messages must be confirmed before submission.

## 5. Recommendation

I recommend piloting the published fully fine-tuned DistilBERT model for ticket routing, with human review of uncertain predictions. Its superior validation, test and independent-message performance supports this choice. I would switch to TF-IDF if a representative real-ticket evaluation showed comparable routing quality at substantially lower operating cost.

## 6. Honest limitation

The experiment cannot establish production accuracy: generated tickets may share wording across splits, only three seeds were evaluated, and the independent messages are not verified real customer tickets. A larger, independently labeled real-world dataset is needed before enabling automatic routing.

Evidence: [validation summary](reports/summary.csv), [all comparison runs](reports/results.csv), [final evaluation](reports/final.json).
