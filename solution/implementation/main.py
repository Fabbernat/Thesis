# C:\PycharmProjects\Peternity\solution\main.py
from config import Config
from solution.data_loader import load_wic_data
from similarity import compute_sentence_similarity
from solution.results.wic_evaluation import evaluate


def main():
    config = Config()

    # Load data and compute similarities
    data, labels = load_wic_data(config.data_file, config.gold_file)
    similarities = compute_sentence_similarity(data)

    # Evaluate model with all metrics
    metrics = evaluate(
        similarities,
        labels,
        data,
        threshold=config.similarity_threshold,
        verbose=True
    )


if __name__ == '__main__':
    main()