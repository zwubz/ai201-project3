# Project Planning: WallStreetBets Classifier

## 1. Community Selection
* **Chosen Community:** `r/wallstreetbets` (WSB)
* **Rationale & Discourse Variety:** WSB is a highly active and distinct online stock trading forum where retail investors discuss options trading and other investments. It fits for text classification because the discourse is highly varied, combining:
  1. **Technical & Mechanical Analysis:** Deep research into options plumbing, market makers, float sizes, index additions, and macroeconomics.
  2. **High-Risk Financial Gambles:** Pure risk, high amounts of gains or loss, users showcase massive active positions purely based on gut feeling or momentum.
  3. **Social Hype & Memes:** Anecdotes, jokes, self-deprecating culture, and casual discussions about trading psychology.
  
  The language uses unique slang, and the structural overlap between bragging, asking questions, and analyzing market events creates a challenging and interesting classification boundary.

---

## 2. Taxonomy & Label Definitions

### Label 1: Due Diligence (DD) & Market Mechanics Analysis
* **Definition:** Posts that present detailed, structured research, fundamental or technical analysis, or mechanical market catalysts (e.g., float size, index inclusion, tariffs, lockups) to argue a specific investment thesis.
* **Example Post 1 (Title/Author):** *“Bullish thesis for SPCX into the summer”* by **Fun_Paleontologist_2**
  * *Content:* Detailed breakdown of CRSP/Vanguard index tracking timelines, option gamma feedback loops, and lockup periods.
* **Example Post 2 (Title/Author):** *“1M+ bet on T1 Energy (TE) PT 2”* by **Beneficial-Ad-7771**
  * *Content:* Analysis of Section 232 solar tariffs, domestic cell production capacity, and the strategic value of the KORE acquisition for data centers.

### Label 2: YOLO & Position Flexing
* **Definition:** Posts primarily focused on showcasing a high-risk trade, massive bet, or significant realized/unrealized gains/losses (often with screenshots or specific options positions) with minimal to no analytical backing.
* **Example Post 1 (Title/Author):** *“100k SPY 6/16 744c YOLO held through iran deal / Trump's birthday weekend”* by **DesktopSurfer**
  * *Content:* Celebrates a $109k SPY options gamble made on a whim, focusing purely on luck and the emotional roller-coaster of the trade.
* **Example Post 2 (Title/Author):** *“5k-66k in 3 days. Life finds a way”* by **lococommotion**
  * *Content:* A pure profit-flexing post bragging about 10x-ing an account through rapid SPY 0DTE scalping in less than 30 seconds with no strategy details.

### Label 3: Discussion & Speculative Query
* **Definition:** Posts that ask questions, discuss market news, or share community-specific culture, memes, and anecdotes without presenting a deep analytical thesis or showcasing a major active trade.
* **Example Post 1 (Title/Author):** *“Attractive women are starting to approach me. God help us all.”* by **Papa_Hoch**
  * *Content:* A humorous, self-deprecating story about a bar interaction where a woman mistook a SpaceX allocation request screenshot.
* **Example Post 2 (Title/Author):** *“Why Adobe?”* by **TenkaiRyo**
  * *Content:* A simple, two-sentence question asking why Adobe is struggling compared to other subscription businesses.

---

## 3. Hard Edge Cases & Resolution Strategies
1. **The "DD + YOLO" Overlap:**
   * *The Ambiguity:* A user posts a massive active position (e.g., "$200k in calls") but also writes a lengthy, structured analysis detailing why the stock will moon (float metrics, news, catalysts).
   * *Resolution Strategy:* We apply a **Hierarchy of Intent**. If the post contains *original, structured data or calculations* (such as specific index inclusion dates, regulatory rulings, or revenue models) that support the thesis, it is classified as **Due Diligence (DD)**. If the text is primarily simple sentiment, hype, or summary of popular community tropes (e.g. "Elon is a genius, short squeeze coming") to justify the screenshot, it is classified as **YOLO & Position Flexing**.
2. **The "DD + Discussion" Overlap:**
   * *The Ambiguity:* A post discussing market news (e.g., an IPO or acquisition) that lists a few dates or numbers but doesn't build a formal argument or buy/sell thesis.
   * *Resolution Strategy:* To qualify as **Due Diligence (DD)**, the post must take a clear, reasoned stance on future price action backed by the data. If the post merely compiles news articles, asks the community what they think, or presents sector trivia without a structured thesis, it is classified as **Discussion & Speculative Query**.
3. **Satirical or Humorous "DD":**
   * *The Ambiguity:* A post that adopts the formal structure of a DD (sections, bullet points, references to economics) but uses joke arguments, anecdotes, or superstitions (e.g., dating patterns or bird droppings) as its data.
   * *Resolution Strategy:* Because the "data" is satirical/anecdotal and not objective market or financial data, these posts are classified as **Discussion & Speculative Query under the humor/memes clause.**

---

## 4. Data Collection & Annotation Plan
* **Source:** We will scrape `old.reddit.com/r/wallstreetbets` using a BeautifulSoup scraper.
* **Target Size:** A minimum of 200 total annotated posts, targeting roughly 60–70 examples per label to maintain a balanced dataset.
* **Mitigating Class Imbalance:** Discussion posts naturally dominate the subreddit feed, while high-quality DD and verified YOLOs are less frequent. If a label is underrepresented:
  1. We will filter the scraping target using Reddit search queries containing specific keywords (e.g. searching for `"YOLO"`, `"loss"`, `"gain"`, or `"position"` for Label 2; and `"DD"`, `"thesis"`, `"catalyst"`, or `"EBITDA"` for Label 1).
  2. We will scrape posts filtered by subreddit flairs (e.g. selecting only posts flaired as `YOLO` or `DD` on the desktop web interface).
  3. We will manually review and down-sample the overrepresented "Discussion" class to ensure a balanced annotation set.

---

## 5. Evaluation Metrics
The distribution of classes is naturally skewed, and different misclassifications carry different operational costs. The following metrics will be used:
1. **Confusion Matrix:** To track exactly which categories are being misclassified (e.g. verifying that DD is not being conflated with Discussion).
2. **Precision (for Due Diligence):** 
   $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$
   * *Why:* If a trader uses this tool to find high-quality research, spamming them with casual discussions or simple memes mislabeled as DD ruins utility. We must minimize False Positives for the DD class.
3. **Recall (for YOLO):**
   $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
   * *Why:* For community alerts or order tracking, we want to capture as many active bets/YOLOs as possible. We want to minimize False Negatives for the YOLO class.
4. **Macro F1-Score:** The harmonic mean of precision and recall averaged equally across all classes, serving as the primary overall metric to verify balanced performance across all three categories.

---

## 6. Definition of Success
The classifier must achieve the following specific, objective criteria on a held-out test set of 50 manually labeled posts:
1. **Macro F1-Score:** $\ge 0.82$ across all three classes.
2. **DD Class Precision:** $\ge 0.85$ (fewer than 15% of posts flagged as DD can be casual discussion or pure memes).
3. **YOLO Class Recall:** $\ge 0.80$ (capturing at least 80% of actual YOLO positions).
4. **Inference Latency:** $< 1.5$ seconds per post when making API calls to Groq, ensuring it can handle real-time streaming ingestion.

## AI Tool Plan
Label stress testing: Gemini 3.5 Flash to generate 5-10 synthetic posts that sit right on the boundary between labels (e.g., posts that look like detailed research but contain a massive YOLO) to test and resolve ambiguities.
Annotation assistance: AI will write a program to scraped from `old.reddit.com/r/wallstreetbets` using a BeautifulSoup scraper targeting recent top posts. Groq API prompt will pre-label bunch of examples before self reviewing. However, I'm not confident in Groq's labels so I manually reviewed then correct with Gemini 3.5 Flash against the rules to establish a higher confidence in annotation.
Failure analysis: Automatic scripting to extract all misclassified posts from test file and use it to detec the failure boundaries.