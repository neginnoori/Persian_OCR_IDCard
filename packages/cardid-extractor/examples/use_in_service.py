"""Run inside the same Python environment as your application and final OCR function."""
from cardid import CardExtractor, ExtractorConfig

# Create once per service process; this instance reuses its loaded GPU models.
extractor = CardExtractor(ExtractorConfig(cache_dir="./cardid-model-cache"))


def prepare_for_ocr(input_image):
    """Return an upright, perspective-corrected PIL RGB card or raise CardIDError."""
    return extractor.extract(input_image)


# Your application controls when to call the function:
# card = prepare_for_ocr(uploaded_image_bytes)
# text = your_persian_ocr(card)  # or np.asarray(card), depending on your OCR interface.
