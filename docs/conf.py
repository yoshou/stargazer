import os

project = 'Stargazer'
copyright = '2024, Stargazer Contributors'
author = 'Stargazer Contributors'

extensions = [
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

breathe_projects = {
    'stargazer': os.path.join(os.path.dirname(__file__), '_doxygen/xml'),
}
breathe_default_project = 'stargazer'

html_theme = 'sphinx_rtd_theme'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
