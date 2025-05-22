# C:\PycharmProjects\Peternity\solution\implementation\main.py
from config import Config
from config import load_wic_data
from similarity import compute_sentence_similarity
from solution.results.wic_evaluation import evaluate


def main():
    config = Config()

    # Load data and compute similarities
    data, labels = load_wic_data(config.data_file, config.gold_file)
    similarities = compute_sentence_similarity(data)

    # Evaluate the model with all metrics
    evaluate(
        similarities,
        labels,
        data,
        threshold=config.similarity_threshold,
        verbose=True
    )


if __name__ == '__main__':
    main()