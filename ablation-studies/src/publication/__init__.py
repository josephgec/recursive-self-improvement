"""Publication-quality output generation."""

from src.publication.latex_tables import LaTeXTableGenerator
from src.publication.figures import PublicationFigureGenerator
from src.publication.significance_stars import add_stars
from src.publication.narrative import NarrativeGenerator
from src.publication.appendix import AppendixGenerator

__all__ = [
    "LaTeXTableGenerator",
    "PublicationFigureGenerator",
    "add_stars",
    "NarrativeGenerator",
    "AppendixGenerator",
]
