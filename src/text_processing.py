"""
Text processing module for cognitive decline detection.
Extracts linguistic features from speech transcripts that may indicate cognitive impairment.
"""

import re
import nltk
import pandas as pd
import numpy as np
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")
    STOPWORDS = set()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Spacy model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    nlp = None

class TextProcessor:
    """Class for processing text transcripts and extracting cognitive decline indicators."""
    
    def __init__(self):
        """Initialize the TextProcessor."""
        # Fillers/hesitation markers
        self.hesitation_markers = {
            'um', 'uh', 'er', 'ah', 'like', 'you know', 'well', 'so', 'hmm',
            'mmm', 'eh', 'uhm', 'umm', 'mm', 'hm', 'erm'
        }
        
        # Common word-finding difficulties phrases
        self.word_finding_phrases = {
            'what do you call it', 'what is it called', 'i can\'t remember the word',
            'i forgot the word', 'what\'s the word', 'you know what i mean', 
            'it\'s on the tip of my tongue', 'what\'s that thing', 'that thing',
            'what\'s it called', 'whatchamacallit', 'what do you call that'
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_hesitation_features(self, text: str) -> Dict[str, float]:
        """
        Extract features related to hesitations and filler words.
        
        Args:
            text: Preprocessed transcript text
            
        Returns:
            Dictionary with hesitation-related features
        """
        if not text:
            return {
                "hesitation_count": 0,
                "hesitation_ratio": 0
            }
        
        # Tokenize text
        tokens = word_tokenize(text)
        
        # Count hesitation markers
        hesitation_count = 0
        for word in tokens:
            if word.lower() in self.hesitation_markers:
                hesitation_count += 1
        
        # Multi-word hesitation phrases
        for phrase in self.hesitation_markers:
            if ' ' in phrase and phrase in text:
                hesitation_count += text.count(phrase)
        
        # Calculate ratio
        total_words = len(tokens)
        hesitation_ratio = hesitation_count / total_words if total_words > 0 else 0
        
        return {
            "hesitation_count": hesitation_count,
            "hesitation_ratio": hesitation_ratio
        }
    
    def extract_word_finding_features(self, text: str) -> Dict[str, float]:
        """
        Extract features related to word-finding difficulties.
        
        Args:
            text: Preprocessed transcript text
            
        Returns:
            Dictionary with word-finding related features
        """
        if not text:
            return {
                "word_finding_difficulty_count": 0,
                "word_finding_difficulty_ratio": 0
            }
        
        # Count word-finding phrases
        difficulty_count = 0
        for phrase in self.word_finding_phrases:
            difficulty_count += text.count(phrase)
        
        # Calculate ratio to sentence count
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)
        difficulty_ratio = difficulty_count / sentence_count if sentence_count > 0 else 0
        
        return {
            "word_finding_difficulty_count": difficulty_count,
            "word_finding_difficulty_ratio": difficulty_ratio
        }
    
    def extract_sentence_complexity(self, text: str) -> Dict[str, float]:
        """
        Extract features related to sentence complexity and structure.
        
        Args:
            text: Preprocessed transcript text
            
        Returns:
            Dictionary with sentence complexity features
        """
        if not text or nlp is None:
            return {
                "avg_sentence_length": 0,
                "avg_word_length": 0,
                "type_token_ratio": 0,
                "noun_verb_ratio": 0,
                "syntactic_complexity": 0,
                "clause_count_avg": 0
            }
        
        # Parse with spaCy for deeper linguistic analysis
        doc = nlp(text)
        
        # Sentence measures
        sentences = list(doc.sents)
        if not sentences:
            return {
                "avg_sentence_length": 0,
                "avg_word_length": 0,
                "type_token_ratio": 0,
                "noun_verb_ratio": 0,
                "syntactic_complexity": 0,
                "clause_count_avg": 0
            }
        
        # Average sentence length in words
        sentence_lengths = [len([token for token in sent if not token.is_punct]) for sent in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        
        # Average word length
        word_lengths = [len(token.text) for token in doc if token.is_alpha]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0
        
        # Type-token ratio (lexical diversity)
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        unique_tokens = set(tokens)
        type_token_ratio = len(unique_tokens) / len(tokens) if tokens else 0
        
        # Part of speech analysis
        noun_count = len([token for token in doc if token.pos_ in ('NOUN', 'PROPN')])
        verb_count = len([token for token in doc if token.pos_ == 'VERB'])
        noun_verb_ratio = noun_count / verb_count if verb_count > 0 else 0
        
        # Syntactic complexity (average dependency tree depth)
        tree_depths = []
        for sent in sentences:
            # For each token, calculate its depth in the dependency tree
            token_depths = []
            for token in sent:
                depth = 0
                current = token
                while current.head != current:  # While not the root
                    depth += 1
                    current = current.head
                token_depths.append(depth)
            if token_depths:
                tree_depths.append(max(token_depths))
        
        avg_tree_depth = np.mean(tree_depths) if tree_depths else 0
        
        # Estimate clause count by counting verbs per sentence
        clause_counts = []
        for sent in sentences:
            verb_count = sum(1 for token in sent if token.pos_ == 'VERB')
            clause_counts.append(verb_count)
        
        avg_clause_count = np.mean(clause_counts) if clause_counts else 0
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length,
            "type_token_ratio": type_token_ratio,
            "noun_verb_ratio": noun_verb_ratio,
            "syntactic_complexity": avg_tree_depth,
            "clause_count_avg": avg_clause_count
        }
    
    def extract_coherence_features(self, text: str) -> Dict[str, float]:
        """
        Extract features related to discourse coherence.
        
        Args:
            text: Preprocessed transcript text
            
        Returns:
            Dictionary with coherence-related features
        """
        if not text:
            return {
                "pronoun_noun_ratio": 0,
                "topic_consistency": 0,
                "repetition_rate": 0
            }
        
        # Parse with spaCy
        doc = nlp(text) if nlp is not None else None
        
        if doc is None:
            return {
                "pronoun_noun_ratio": 0,
                "topic_consistency": 0,
                "repetition_rate": 0
            }
        
        # Pronoun to noun ratio (high values may indicate referential issues)
        pronoun_count = len([token for token in doc if token.pos_ == 'PRON'])
        noun_count = len([token for token in doc if token.pos_ in ('NOUN', 'PROPN')])
        pronoun_noun_ratio = pronoun_count / noun_count if noun_count > 0 else 0
        
        # Word repetition (excluding stopwords)
        content_tokens = [token.lemma_.lower() for token in doc 
                         if token.is_alpha and not token.is_stop and len(token.text) > 2]
        
        if not content_tokens:
            return {
                "pronoun_noun_ratio": pronoun_noun_ratio,
                "topic_consistency": 0,
                "repetition_rate": 0
            }
        
        # Count occurrences of each token
        token_counts = {}
        for token in content_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Calculate repetition rate (proportion of content words that appear multiple times)
        repeated_tokens = sum(1 for count in token_counts.values() if count > 1)
        repetition_rate = repeated_tokens / len(token_counts) if token_counts else 0
        
        # Simple topic consistency measure: concentration of content words
        # Higher values suggest more focused discourse
        content_word_freq = {}
        for token in content_tokens:
            content_word_freq[token] = content_word_freq.get(token, 0) + 1
        
        if not content_word_freq:
            topic_consistency = 0
        else:
            # Calculate entropy of content word distribution (lower entropy = more consistency)
            total = sum(content_word_freq.values())
            probs = [count/total for count in content_word_freq.values()]
            entropy = -sum(p * np.log(p) for p in probs)
            
            # Normalize and invert so higher values mean more consistency
            topic_consistency = 1 / (1 + entropy) if entropy > 0 else 1
        
        return {
            "pronoun_noun_ratio": pronoun_noun_ratio,
            "topic_consistency": topic_consistency,
            "repetition_rate": repetition_rate
        }
    
    def extract_all_features(self, transcript: str) -> Dict[str, Any]:
        """
        Extract all text features for cognitive decline detection.
        
        Args:
            transcript: Raw transcript text
            
        Returns:
            Dictionary containing all extracted features
        """
        if not transcript:
            logger.warning("Empty transcript provided")
            return {}
        
        try:
            # Preprocess text
            preprocessed_text = self.preprocess_text(transcript)
            
            # Extract all feature sets
            hesitation_features = self.extract_hesitation_features(preprocessed_text)
            word_finding_features = self.extract_word_finding_features(preprocessed_text)
            complexity_features = self.extract_sentence_complexity(preprocessed_text)
            coherence_features = self.extract_coherence_features(preprocessed_text)
            
            # Combine all features
            all_features = {
                **hesitation_features,
                **word_finding_features,
                **complexity_features,
                **coherence_features
            }
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {}

    def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze transcript and return features and interpretations.
        
        Args:
            transcript: Raw transcript text
            
        Returns:
            Dictionary with features and human-readable interpretations
        """
        features = self.extract_all_features(transcript)
        
        # Add simple interpretations
        interpretations = {}
        
        if features:
            # Interpret hesitation patterns
            if features["hesitation_ratio"] > 0.05:
                interpretations["hesitations"] = "High frequency of hesitation markers, may indicate word-finding difficulties"
            else:
                interpretations["hesitations"] = "Normal frequency of hesitation markers"
            
            # Interpret word-finding difficulties
            if features["word_finding_difficulty_ratio"] > 0.1:
                interpretations["word_finding"] = "Exhibits multiple instances of word-finding difficulties"
            else:
                interpretations["word_finding"] = "Few or no explicit word-finding difficulties detected"
            
            # Interpret sentence complexity
            if features["avg_sentence_length"] < 5:
                interpretations["sentence_complexity"] = "Very simple, short sentences which may indicate cognitive difficulties"
            elif features["avg_sentence_length"] < 10:
                interpretations["sentence_complexity"] = "Somewhat simplified sentence structure"
            else:
                interpretations["sentence_complexity"] = "Normal sentence complexity"
            
            # Interpret coherence
            if features["topic_consistency"] < 0.3:
                interpretations["coherence"] = "Low topic consistency, potentially indicating disorganized thinking"
            else:
                interpretations["coherence"] = "Normal topic consistency and coherence"
        
        return {
            "features": features,
            "interpretations": interpretations
        }
