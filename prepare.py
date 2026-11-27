"""Fixed split and stratified training samples."""
from sklearn.model_selection import train_test_split

from tickets_data import make_tickets

SPLIT_SEED = 42
SIZES = (50, 100, 300)
SEEDS = (42, 123, 2026)


def split_tickets(seed=SPLIT_SEED):
    """Return disjoint 60/20/20 partitions of the unchanged data."""
    data = make_tickets()
    train, rest = train_test_split(
        data, test_size=0.4, random_state=seed, stratify=data.label
    )
    val, test = train_test_split(
        rest, test_size=0.5, random_state=seed, stratify=rest.label
    )
    return train, val, test


def sample_train(train, n, seed):
    """Select exactly n rows from train only."""
    if n == len(train):
        return train.sample(frac=1, random_state=seed)
    sample, _ = train_test_split(
        train, train_size=n, random_state=seed, stratify=train.label
    )
    return sample


if __name__ == '__main__':
    for name, frame in zip(('train', 'validation', 'test'), split_tickets(), strict=True):
        print(name, len(frame), frame.label.value_counts().to_dict())
