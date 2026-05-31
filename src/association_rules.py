import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules as mlxtend_association_rules
from mlxtend.preprocessing import TransactionEncoder

from src.keyword_frequency import SKILL_KEYWORDS, _skill_matches
from src.preprocessing import clean_text


def _empty_rules_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["antecedents", "consequents", "support", "confidence", "lift"]
    )


def extract_skill_transactions(descriptions):
    transactions = []

    for description in descriptions:
        cleaned_description = clean_text(description)
        if cleaned_description == "":
            continue

        skills = [
            skill
            for skill in SKILL_KEYWORDS
            if _skill_matches(skill, cleaned_description)
        ]

        if skills:
            transactions.append(sorted(skills))

    return transactions


def mine_skill_rules(transactions, min_support=0.05, min_confidence=0.3):
    if not transactions:
        return _empty_rules_df()

    encoder = TransactionEncoder()
    encoded_transactions = encoder.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(encoded_transactions, columns=encoder.columns_)

    frequent_itemsets = apriori(
        transaction_df,
        min_support=min_support,
        use_colnames=True,
    )

    if frequent_itemsets.empty:
        return _empty_rules_df()

    rules_df = mlxtend_association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    if rules_df.empty:
        return _empty_rules_df()

    return rules_df[["antecedents", "consequents", "support", "confidence", "lift"]]


def format_rules(rules_df, top_n=10):
    if rules_df.empty:
        return _empty_rules_df()

    formatted_df = rules_df.copy()
    formatted_df["antecedents"] = formatted_df["antecedents"].apply(
        lambda skills: ", ".join(sorted(skills))
    )
    formatted_df["consequents"] = formatted_df["consequents"].apply(
        lambda skills: ", ".join(sorted(skills))
    )

    formatted_df = formatted_df.sort_values(
        by=["lift", "confidence"],
        ascending=False,
    )

    return formatted_df[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].head(top_n)
