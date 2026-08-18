import pandas as pd

df = pd.read_csv("CFPB_Complaints_May_July_2026.csv.csv")

print(df.head())
print(df.columns.tolist())
print("\nCompany response outcomes:")
print(df["Company response to consumer"].value_counts())

print("\nTimely response:")
print(df["Timely response?"].value_counts())
print("\nTimely response by company response outcome:")
print(
    pd.crosstab(
        df["Company response to consumer"],
        df["Timely response?"]
    )
)
print("\nTimely response rate within each company response outcome:")

timely_rates = pd.crosstab(
    df["Company response to consumer"],
    df["Timely response?"],
    normalize="index"
) * 100

print(timely_rates.round(2))

print("\nResolution outcomes by product:")

resolution_by_product = pd.crosstab(
    df["Product"],
    df["Company response to consumer"]
)

print(resolution_by_product)
print("\nResolution outcome percentage by product:")

resolution_rates = pd.crosstab(
    df["Product"],
    df["Company response to consumer"],
    normalize="index"
) * 100

print(resolution_rates.round(2))

from scipy.stats import chi2_contingency

chi2, p_value, degrees_of_freedom, expected = chi2_contingency(
    resolution_by_product
)

print("\nChi-square test:")
print("Chi-square statistic:", round(chi2, 2))
print("Degrees of freedom:", degrees_of_freedom)
print("P-value:", p_value)
import numpy as np

n = resolution_by_product.to_numpy().sum()

cramers_v = np.sqrt(
    chi2 / (n * min(
        resolution_by_product.shape[0] - 1,
        resolution_by_product.shape[1] - 1
    ))
)

print("\nCramer's V:", round(cramers_v, 3))

print("\nPearson residuals:")

residuals = (
    resolution_by_product - expected
) / np.sqrt(expected)

residuals_df = pd.DataFrame(
    residuals,
    index=resolution_by_product.index,
    columns=resolution_by_product.columns
)

print(residuals_df.round(2))
print("\nCredit card sub-issues:")
print(
    df.loc[df["Product"] == "Credit card", "Sub-issue"]
    .value_counts()
    .head(10)
)
print("\nCredit card purchase dispute narratives:")

dispute_narratives = df.loc[
    df["Sub-issue"] == "Credit card company isn't resolving a dispute about a purchase on your statement",
    "Consumer complaint narrative"
].dropna()

print("Number of narratives available:", len(dispute_narratives))
print("\nSample narratives:")
print(dispute_narratives.head(5).to_string(index=False))

from sklearn.feature_extraction.text import TfidfVectorizer

print("\nTF-IDF analysis:")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=30,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(dispute_narratives)

terms = vectorizer.get_feature_names_out()
scores = tfidf_matrix.mean(axis=0).A1

tfidf_results = pd.DataFrame({
    "Term": terms,
    "TF-IDF Score": scores
}).sort_values(
    "TF-IDF Score",
    ascending=False
)

print(tfidf_results.head(20).to_string(index=False))

import re

print("\nCleaning narratives...")

clean_narratives = dispute_narratives.str.lower()

# Remove redacted/anonymized text such as XXXX or XX
clean_narratives = clean_narratives.str.replace(
    r'\b[xX]+\b',
    ' ',
    regex=True
)

# Remove numbers
clean_narratives = clean_narratives.str.replace(
    r'\d+',
    ' ',
    regex=True
)

# Remove punctuation
clean_narratives = clean_narratives.str.replace(
    r'[^a-z\s]',
    ' ',
    regex=True
)

# Remove extra spaces
clean_narratives = clean_narratives.str.replace(
    r'\s+',
    ' ',
    regex=True
).str.strip()

print("Narratives after cleaning:", len(clean_narratives))
print("\nSample cleaned narratives:")
print(clean_narratives.head(5).to_string(index=False))
print("\nTF-IDF analysis after cleaning:")

vectorizer_clean = TfidfVectorizer(
    stop_words="english",
    max_features=30,
    ngram_range=(1, 2)
)

tfidf_matrix_clean = vectorizer_clean.fit_transform(clean_narratives)

terms_clean = vectorizer_clean.get_feature_names_out()
scores_clean = tfidf_matrix_clean.mean(axis=0).A1

tfidf_results_clean = pd.DataFrame({
    "Term": terms_clean,
    "TF-IDF Score": scores_clean
}).sort_values(
    "TF-IDF Score",
    ascending=False
)

print(tfidf_results_clean.head(20).to_string(index=False))

print("\nNarratives by resolution outcome:")

dispute_data = df.loc[
    df["Sub-issue"] == "Credit card company isn't resolving a dispute about a purchase on your statement",
    ["Consumer complaint narrative", "Company response to consumer"]
].dropna(subset=["Consumer complaint narrative"])

print(
    dispute_data["Company response to consumer"]
    .value_counts()
)
print("\nComparing monetary relief vs. explanation narratives:")

# Create a dataset containing the cleaned narratives and resolution outcome
narrative_analysis = pd.DataFrame({
    "narrative": clean_narratives,
    "outcome": df.loc[clean_narratives.index, "Company response to consumer"]
})

# Keep only explanation and monetary relief
narrative_analysis = narrative_analysis[
    narrative_analysis["outcome"].isin([
        "Closed with explanation",
        "Closed with monetary relief"
    ])
]

# Create one TF-IDF model across both groups
comparison_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=100
)

comparison_matrix = comparison_vectorizer.fit_transform(
    narrative_analysis["narrative"]
)

comparison_terms = comparison_vectorizer.get_feature_names_out()

comparison_df = pd.DataFrame(
    comparison_matrix.toarray(),
    columns=comparison_terms,
    index=narrative_analysis.index
)

# Calculate average TF-IDF score for each outcome
comparison_df["Outcome"] = narrative_analysis["outcome"]

group_means = comparison_df.groupby("Outcome").mean(numeric_only=True).T

# Calculate the difference between monetary relief and explanation
group_means["Difference"] = (
    group_means["Closed with monetary relief"]
    - group_means["Closed with explanation"]
)

print("\nTerms more associated with monetary-relief complaints:")

print(
    group_means
    .sort_values("Difference", ascending=False)
    .head(15)
    .round(4)
)

print("\nTerms more associated with explanation complaints:")

print(
    group_means
    .sort_values("Difference", ascending=True)
    .head(15)
    .round(4)
)
