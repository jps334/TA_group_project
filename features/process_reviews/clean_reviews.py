from features.process_text.clean import clean_text
import copy

def clean_review(review):
    """Cleans text of review."""
    cleaned_review = copy.copy(review)
    cleaned_review.cleaning_log, cleaned_review.reviewtext_cleaned = clean_text(cleaned_review.reviewText, tokenize=True)
    return cleaned_review


