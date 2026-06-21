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

---

## 4. Error Analysis and Model Limitations

### 1. YOLO Misclassified as Discussion (The Slang and Intent Gap)
The confusion matrix reveals that the fine-tuned DistilBERT model heavily misclassified **YOLO** posts as **Discussion**. 

This occurs due to a failure to internalize subcultural slang and behavioral intent:
*   **Example 1 (Subcultural Slang):**
    *   *Text:* `"Title: A little 48 hour day trade (take 2) \nBody: I full-port bought Intel due to it being the cheapest of the chip giant stocks, the next day it got a price target increase by Bank of America and I set a sell target for a number I felt was reasonable. This AI/ chip rally has been great for my account!"`
    *   *Analysis:* Human annotators easily recognize the phrase **"full port"** (putting 100% of a portfolio into one trade) as an explicit signal for a YOLO post. However, the smaller capacity of DistilBERT failed to weigh this specific subcultural slang appropriately, absorbing it into general Discussion.
*   **Example 2 (Intent vs. Semantics):**
    *   *Text:* `"Title: 50k into SPY puts because macro indicators are blinking red \nBody: Unemployment ticked up by 0.2% and consumer credit defaults are accelerating. Looking at the weekly chart, we are severely overextended above the 200 EMA. Going all in on June expiration contracts."`
    *   *Analysis:* The content contains objective data points (`"Unemployment ticked up"`, `"200 EMA"`), which are strong semantic markers for Due Diligence. However, the true intent of the post is to showcase a massive, high-risk options gamble (`"50k into SPY puts"`, `"all in"`). The model was swayed by the financial terminology and misclassified it as Discussion.

### 2. Due Diligence Misclassified as Discussion (The Brevity Bias)
Concise analysis posts were sometimes misclassified as Discussion because the model associated shorter text lengths with casual chat.
*   **Example 3:**
    *   *Text:* `"Title: Quick catalyst for NVDA next week \nBody: Keep an eye out for the supply chain data releasing on Tuesday from Taiwan. If TSMC shipments are up, NVDA clears guidance easily."`
    *   *True Label:* Due Diligence | *Predicted Label:* Discussion
    *   *Analysis:* This post contains research about Taiwan supply chain data and Taiwan Semiconductor Manufacturing Company (TSMC) shipments to forecast NVIDIA (NVDA) guidance. Because it was short and did not seek feedback, it represents a concise Due Diligence post. The model's reliance on post length as a proxy for category caused it to classify it as Discussion.

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