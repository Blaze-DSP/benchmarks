"""Common utility functions for benchmarks."""


def print_header(title: str, width: int = 70) -> None:
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_footer(width: int = 70) -> None:
    """Print a formatted footer."""
    print("=" * width)

