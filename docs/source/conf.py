import datetime
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.parent.parent.absolute().as_posix())

from lumafit import __version__

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lumafit"
author = "Mirza Arnaut"
copyright = f"{datetime.datetime.now().year}, {author}"
version = __version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Standard Sphinx extension for docstrings
    "sphinx.ext.napoleon",  # For parsing Google/NumPy style docstrings
    "sphinx_autodoc_typehints",  # For type hints
    "autoapi.extension",  # For automated API documentation
    "myst_parser",  # For using Markdown files
    "sphinx_copybutton",  # For adding a copy button to code blocks
    "sphinxcontrib.bibtex",  # For bibliography/citations
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "restructuredtext"}

# -- Extension configurations ------------------------------------------------

# AutoAPI configuration
# Specify the directories containing the Python code to document.
# This should point to your source package(s).
autoapi_dirs = ["../../lumafit"]
autoapi_type = "python"
autoapi_add_toctree_entry = True  # Add generated API docs to the TOC
autoapi_generate_api = True  # Generate individual pages for modules, classes, etc.
# Optionally skip certain members (e.g., private ones)
autoapi_python_class_content = "both"  # Include docstrings for class and __init__
autoapi_member_order = "bysource"  # Or 'alphabetical'
autoapi_options = [
    "members",
    "imported-members",
    "private-members",
    # "undoc-members",
    # "show-inheritance",
    # "show-module-members",
    # "show-module-summary",
]
# autoapi_keep_files = True # Keep the generated .rst files for inspection

# MyST Parser configuration
# Configure how MyST handles markdown files (optional extensions)
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
# Set the default role for single backticks (e.g., `object`)
# 'py:obj' is common for Python objects
# This makes `MyClass` equivalent to :py:obj:`MyClass`
# default_role = 'py:obj'

# Bibtex configuration
# The path to your bibliography file(s), relative to the source directory
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# Napoleon configuration (usually works fine with defaults)
# napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False  # Include __init__ docstrings
napoleon_include_private_with_doc = True  # Usually False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True
napoleon_preprocess_types = False  # Let sphinx-autodoc-typehints handle types
napoleon_type_aliases = None
napoleon_attr_annotations = True


# sphinx-autodoc-typehints configuration (usually works fine with defaults)
# Add parameter types from Napoleon processing
always_document_param_types = True
# Show short names for types (e.g. ndarray instead of numpy.ndarray)
typehints_fully_qualified = False
# Process return type hints
typehints_document_rtype = True
# Don't use napoleon rtype processing, let extension handle it
# typehints_use_rtype = False
# Show default values after comma, 'braces' is other option
# typehints_defaults = "comma"
# Optional: Simplify representation of complex types like Union[str, Path]
# typehints_formatter = lambda annotation, config: repr(annotation)
always_use_bars_union = True

# autodoc
autoclass_content = "class"
autodoc_typehints = "none"
autodoc_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
}
inheritance_alias = {}

# sphinx-copybutton configuration
# copybutton_prompt_text = "$ " # Add a prompt pattern to remove from copied text

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = []
# html_show_sourcelink = True
html_context = dict(
    github_user="arunoruto",
    github_repo="lumafit",
    github_version="main",
    doc_path="docs/source/",
)

# Customize theme options here (examples)
# html_theme_options = {
#     "sidebar_hide_name": False,
#     "navigation_depth": 4,
#     "analytics": {
#         "google_analytics_id": "G-XXXXXXXXXX",
#     },
#     "repository_url": "https://github.com/your_username/lumafit", # Update this
#     "use_repository_button": True,
#     "use_issues_button": True,
#     "use_edit_page_button": True,
# }

# PyData Theme Options
html_theme_options = {
    "github_url": f"https://github.com/{html_context['github_user']}/{html_context['github_repo']}",
    # "navigation_depth": 4,
    # "analytics": {
    #     "google_analytics_id": "G-XXXXXXXXXX",  # Update this
    # },
    # "repository_url": "https://github.com/arunoruto/lumafit",
    # "use_repository_button": True,
    # "use_issues_button": True,
    "use_edit_page_button": True,
    # "show_toc_level": 2,
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"  # Or 'friendly', 'colorful', 'monokai', etc.

## Intersphinx Configuration: Set up links to external documentation:
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    # Add others like scipy, pandas if you use/reference them
}
