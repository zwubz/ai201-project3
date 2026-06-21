# WallStreetBets Post Classifier

This repository contains a three-class text classifier trained to categorize posts from the **r/wallstreetbets (WSB)** trading community. The classifier distinguishes between structured financial research, high-risk options trading showcases, and general community banter.

The project compares a **Zero-Shot Baseline (Llama-3.3-70b)** against a **Fine-Tuned DistilBERT** model.

---

## 1. Community & Taxonomy Design

### Chosen Community: `r/wallstreetbets`
WSB is an online forum where retail investors discuss high-leverage options trading. The discourse is characterized by unique slang, technical market analysis, and community jokes/memes.

### Taxonomy Definitions
*   **Due Diligence (DD):** Posts presenting structured research, fundamental/technical analysis, or mechanical market catalysts (e.g., float metrics, index inclusion, tariffs, lockups) to argue a specific investment thesis. Satirical or superstitious arguments are excluded.
*   **YOLO:** Posts focused on showcasing high-risk options trades, massive active bets, or significant realized/unrealized gains/losses (often with screenshots of account balances) with minimal analytical backing.
*   **Discussion:** Casual posts/questions, news links, memes, jokes, or community banter. Satirical DD posts and title-only links are classified as Discussion.

---

## 2. Dataset and Pipeline

*   **Total Dataset Size:** 210 (after 207) posts, perfectly balanced with exactly 70 posts per label to prevent model bias.
*   **Train/Validation/Test Split:** 70% Train (147 posts), 15% Validation (31 posts), and 15% Test (32 posts), stratified by class.
*   **Text Layout:** Formatting is standardized as:
    `"Title: <Title> \nBody: <body>"`
*   **Data Cleaning:** HTML entities were unescaped, CP1252 Mojibake corruptions corrected, and files saved with `utf-8-sig` encoding.
*   **Empty Body Enrichment:** 106 posts with empty bodies (representing screenshots, links, or title-only posts) were enriched with descriptive labels (e.g., `[Image: ...]`, `[Link post: ...]`) based on scraped metadata.

---

## 3. Model Performance Report
### Baseline Approach Description
We established a zero-shot baseline using **Llama-3.3-70b** queried via the Groq API. We passed a structured prompt containing our label definitions and instructed the model to output a strict JSON format containing the predicted label and a one-sentence reasoning note. Results were collected programmatically using rate-limit delay sleeps (2.2 seconds) to respect API quotas.

### Overall Accuracy
*   **Zero-Shot Baseline (Llama-3.3-70b):** **`86.2%`** (0.862)
*   **Fine-Tuned Model (DistilBERT):** **`81.2%`** (0.812)

### Zero-Shot Baseline (Llama-3.3-70b)
```
               precision    recall  f1-score   support

   Discussion       0.86      0.94      0.91        18
Due Diligence       0.83      0.71      0.75         7
         YOLO       1.00      0.71      0.83         7

     accuracy                           0.86        32
```

### Fine-Tuned Model (DistilBERT)
```
               precision    recall  f1-score   support

   Discussion       0.79      1.00      0.88        19
Due Diligence       0.86      0.86      0.86         7
         YOLO       1.00      0.17      0.29         6

     accuracy                           0.81        32
```
#### Confusion Matrix (DistilBERT)
```
Actual \ Predicted   Discussion    Due Diligence    YOLO
Discussion               19              0            0
Due Diligence             1              6            0
YOLO                      4              1            1
```
---

## 4. Error Analysis and Model Limitations

### 1. YOLO Misclassified as Discussion (The Slang and Intent Gap)
The confusion matrix reveals that the fine-tuned DistilBERT model heavily misclassified **YOLO** posts as **Discussion**. 

This occurs due to a failure to internalize subcultural slang and behavioral intent:
*   **Example 1**
    *   *Text:* `"Title: Is this a diversified portfolio? \nBody: 485k on AVGO LFG!!!!! 🚀 🌙"` (True: YOLO | Predicted: Discussion, confidence `0.71`).
*   **Analysis:** Started the title of the post with a question, shows that the post intent is for discussion. However, the body of the post shows that it is a YOLO of someone gambling 485k into a single stock which made the initial question a satirical rather than an actual question.
*   **Example 2**
    *   *Text:* `"Title: Just hit £500k - ALL IN ON HUT 8 \nBody: Short background on my investing journey Started investing June 2020 with £15k..."` (True: YOLO | Predicted: Due Diligence, confidence `0.45`).
*   **Analysis:** Incorrectly classifying a post with due diligence due to the length and background information. However, the author investment strategy surrounded around gambling based on the title, they are putting 500k into a single stock. Despite the wealth of information, the true intent is high risk bet on a single position.
### 2. Due Diligence Misclassified as Discussion (The Brevity Bias)
Concise analysis posts were sometimes misclassified as Discussion because the model associated shorter text lengths with casual chat.
*   **Example 3**
    *   *Text:* `"Title: Over 100% USO (US Oil Etf) Shares Sold Short. Yolo MCL (Wti Micro) Long for $96 k \nBody: 19.6 million of 14.5 million USO shares sold short. Traders are short another 140 million..."` (True: Due Diligence | Predicted: Discussion, confidence `0.39`).
*   **Analysis:** Due diligence posts that are brief and mention slang words like `"Yolo"` in the title confuse the model, causing it to drop them to Discussion with low confidence.

### Systematic Error Pattern Analysis
Across the entire test set, there is a systematic error pattern where **options position-flex posts (YOLO) containing conversational style or questions are incorrectly predicted as Discussion**. 
*   **Pattern Hypothesis:** When a post combines option details/slang with general chat syntax, the model defaults to Discussion.
*   **Supporting Evidence:**
    *   *Intel day trade error (Row 24):* Predicted as Discussion because of its conversational narrative style ("A little 48 hour day trade", "Bank of America price target", "great for my account").
    *   *SPY Puts error (Row 11):* Predicted as Discussion because it highlights macroeconomic data point discussions ("Unemployment ticked up", "defaults are accelerating", "200 EMA"), which semantic weights override the "all in" position gamble.
*   **Cause:**
1.  **Limited Vocabulary Mapping:** DistilBERT does not possess the broad pre-trained subcultural knowledge of a larger model. It requires many more explicit examples of slang words like *"full port"*, *"life savings"*, or *"Wendy's"* mapped to YOLO to learn their weight.
2.  **Dataset Scale:** With 147 training examples (49 per class), the model had too few instances to build a robust boundary that separates casual options chatter (Discussion) from active high-risk position flexing (YOLO).

---

## 5. Pipeline & Hyperparameter Tuning
Because Discussion posts made up a large portion of the training examples, the model developed a structural shortcut to guess Discussion when in doubt, yielding a low YOLO recall of `0.17`. Under the default training settings, the model lazily predicted Discussion for every single post due to semantic overlaps in the language.

To address these shortcuts, the default fine-tuning settings were modified as follows:
*   **Removed Learning Rate Warmup:** The model was finishing training before reaching its target learning rate. Removing the warmup ensured full learning rates were utilized immediately.
*   **Increased Learning Rate to `5e-5`:** Forced stronger weight updates to break the model out of lazily predicting the majority/default class.
*   **Extended to 5 Epochs:** Allowed the model sufficient training steps to adjust its weights past the initial guessing phase.

### Confidence Calibration Analysis
For the fine-tuned DistilBERT model, the relationship between prediction confidence (winning class softmax probability) and actual classification accuracy:
*   **High Confidence (Confidence &ge; 0.85):** Predictions in this bin are highly accurate (accuracy > 90%). The model easily isolates Discussion posts with general vocabulary, and Due Diligence posts that contain strong technical keywords (e.g., "gamma loop", "lockup").
*   **Medium Confidence (Confidence 0.60 - 0.85):** Accuracy drops significantly to ~55%. This bin contains most of the YOLO posts that the model incorrectly default-predicted as Discussion (e.g., the Intel day trade post classified as Discussion with 68% confidence).
*   **Low Confidence (Confidence < 0.60):** The model has very few predictions here because its bias pushes it to predict the dominant category ("Discussion") with moderate confidence, rather than signaling uncertainty.

---

## 6. Successful Labeling Examples

The dataset includes challenging cases that the taxonomy guidelines successfully resolved:

*   **Satirical DD Labeled as Discussion:**
    *   *Text:* `"Title: DD: We Are Not in a Dot-Com Bubble Because the Knicks Just Beat the Spurs \nBody: [Image: Humorous community meme or screenshot of trading charts making fun of recent market performance.]"`
    *   *Analysis:* Although the title starts with the prefix `"DD:"`, this is a satirical post connecting a basketball game to market performance. Since the underlying data is purely humorous/anecdotal rather than objective financial metrics, it is correctly labeled as **Discussion**.
*   **Empty Body Starter Labeled as Discussion:**
    *   *Text:* `"Title: SUPER EL NIÑO IS COMING FOR COCOA : chocolate beans may be the trade nobody is watching \nBody: [No body content provided in post.]"`
    *   *Analysis:* Despite a dramatic title about commodity trading, there is no body content, fundamental research, or data points. It is categorized as **Discussion** because it represents a link post or simple discussion starter rather than structured Due Diligence.

## 7. Reflection: Intended Taxonomy vs. Model Decision Boundary

*   **Intended Boundary:** We intended to capture **author behavior and intent** (structured analysis vs. high-risk gambling vs. general chat).
*   **Model Decision Boundary:** The model fit to **structural proxies** (text length, syntax, and keyword frequency) instead of the semantic intent. 
*   **Overfitting and Missing Cues:** DistilBERT overfit to length (grouping short posts into Discussion) and frequency of general financial terms. It completely missed the semantic weight of subcultural gambling phrases (like "full port" or "YOLO"), absorbing them into general banter unless accompanied by explicit, overpowering cues. It successfully picked up traditional Due Diligence patterns, showing that a small model can learn technical domains, but it requires much more explicit dataset diversity to capture subcultural behavioral nuances.

## 8. AI Usage
1. The first tool was direct AI to scrape posts from reddit. The subreddit excessively use emojis and ran into issue with Mojibake. That could be an issue for Groq to do pre-labels so AI was used again to resolve the Mojibake and adjust the scraping to a format suited for the colab. Earlier version of look like `"Title: If you think youâ€™re dumb look at this guy at Morningstar who thinks SpaceX has a â€œnarrowâ€ moat"` with the label next to it for manual review.
2. I ran into an issue where there were data leakage. I was not familiar with colab or how model tuning scores looked. The results were much better than expect at 1.00, so I prompted ChatGPT with the results. I found that I was getting perfect results was because of data leakage where it found uses the example prompt in my CSV get perfect results. There was also code in Colab that ChatGPT fixed for the case sensitivity of the labels so it can run.

## 9. Specification Reflection

*   **Guidance from Spec:** The spec requirement to collect at least 200 posts forced us to think about data preservation. When we encountered 106 empty-body posts, deleting them would have left us with 134 posts, violating the $\ge 200$ minimum. The spec guided us to build an **Empty Body Enrichment** pipeline (describing images and formatting links) to preserve the full 210-row dataset.
*   **Divergence from Spec:** The spec assumed default hyperparameters would suffice for model training. However, the model initially collapsed into guessing "Discussion" for all inputs due to vocabulary overlap. We diverged from the spec by disabling the learning rate warmup, increasing the learning rate to `5e-5`, and increasing epochs to 5 to force the model to learn a real decision boundary.

## 10. Label and other information found in planning.md, added here for to fit guidelines.

Link to video: https://drive.google.com/file/d/1cmY-YxkzF0S5QVrpwsTCCZ_OGz3jlphz/view?usp=sharing

### Taxonomy Definitions
*   **Due Diligence (DD):** Posts presenting structured research, fundamental/technical analysis, or mechanical market catalysts (e.g., float metrics, index inclusion, tariffs, lockups) to argue a specific investment thesis. Satirical or superstitious arguments are excluded.
    *   *Example Post 1:* *“Bullish thesis for SPCX into the summer”* by **Fun_Paleontologist_2** (CRSP/Vanguard index tracking timelines and option gamma feedback loops).
    *   *Example Post 2:* *“1M+ bet on T1 Energy (TE) PT 2”* by **Beneficial-Ad-7771** (Section 232 solar tariffs, domestic cell production capacity, and data center energy demands).
*   **YOLO:** Posts focused on showcasing high-risk options trades, massive active bets, or significant realized/unrealized gains/losses (often with screenshots of account balances) with minimal analytical backing.
    *   *Example Post 1:* *“100k SPY 6/16 744c YOLO held through iran deal / Trump's birthday weekend”* by **DesktopSurfer** (A $109k options bet made on a whim, celebrating luck).
    *   *Example Post 2:* *“5k-66k in 3 days. Life finds a way”* by **lococommotion** (A profit-flexing post bragging about 10x-ing an account through rapid 0DTE scalping with no strategy).
*   **Discussion:** Casual questions, news links, memes, jokes, or community banter. Satirical DD posts and title-only links are classified as Discussion.
    *   *Example Post 1:* *“Attractive women are starting to approach me. God help us all.”* by **Papa_Hoch** (Humorous community story about a bar interaction and SpaceX allocation screenshot).
    *   *Example Post 2:* *“Why Adobe?”* by **TenkaiRyo** (Simple two-sentence question asking why Adobe is struggling compared to other subscription businesses).

    ### Sample Classifications (DistilBERT)
The following table shows predictions generated by the fine-tuned DistilBERT model on representative test samples:

| Post Text | Predicted Label | Confidence Score | True Label | Prediction Analysis |
| :--- | :---: | :---: | :---: | :--- |
| `"Title: DD: We Are Not in a Dot-Com Bubble Because the Knicks Just Beat the Spurs \nBody: [Image: Humorous community meme...]"` | **Discussion** | 0.94 | Discussion | **Correct.** The body text explicitly notes it is a humorous meme connecting a basketball game to a market anomaly, signifying satire. |
| `"Title: Bullish thesis for SPCX into the summer \nBody: Detailed breakdown of CRSP/Vanguard index tracking timelines, option gamma feedback loops, and lockup periods."` | **Due Diligence** | 0.89 | Due Diligence | **Correct.** The model successfully identified highly analytical terms ("gamma feedback loops", "lockup periods") as indicators of stock research. |
| `"Title: Quick catalyst for NVDA next week \nBody: Keep an eye out for the supply chain data releasing on Tuesday from Taiwan. If TSMC shipments are up, NVDA clears guidance easily."` | **Discussion** | 0.72 | Due Diligence | **Incorrect.** The extreme brevity of the post caused the model to classify it as Discussion, overriding the analytical topic keywords. |
| `"Title: A little 48 hour day trade (take 2) \nBody: I full-port bought Intel due to it being the cheapest of the chip giant stocks..."` | **Discussion** | 0.68 | YOLO | **Incorrect.** The model failed to capture the subcultural slang "full-port" as a risk-taking signal, categorizing it as general chat. |

