#!/usr/local/bin/python3
#==============================================================================
#     File: notes.py
#  Created: 06/07/2018, 16:51
#   Author: Bernie Roesler
#
"""
  Description: Note code for basic operations.
"""
#==============================================================================

from nltk.corpus import wordnet as wn

synonyms = []
antonyms = []

for syn in wn.synsets("knit"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print('synonyms = ', set(synonyms))
print('antonyms = ', set(antonyms))

#==============================================================================
#==============================================================================
