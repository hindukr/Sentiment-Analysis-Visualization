"""
Advanced Sentiment Analysis System
====================================
Features:
- Multi-method sentiment analysis (VADER, TextBlob, Custom Lexicon)
- Emotion detection using NLP techniques
- Support for Amazon reviews, social media, and news data
- Trend analysis and visualization
- Marketing and product development insights
- Export results to CSV and JSON

Dataset Links:
1. Amazon Fine Food Reviews: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
2. Amazon Reviews (4M): https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
3. Twitter Sentiment: https://www.kaggle.com/datasets/kazanova/sentiment140
4. IMDB Reviews: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Installation:
pip install pandas numpy matplotlib seaborn nltk textblob vaderSentiment wordcloud scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk
    
    # Download required NLTK data
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False
    print("NLTK not available. Install with: pip install nltk")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Install with: pip install textblob")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except:
    WORDCLOUD_AVAILABLE = False


class SentimentAnalyzer:
    """
    Comprehensive Sentiment Analysis System
    """
    
    def _init_(self):
        """Initialize sentiment analyzer with lexicons"""
        
        # Sentiment Lexicons
        self.positive_words = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 'love', 'best',
            'perfect', 'awesome', 'outstanding', 'brilliant', 'superb', 'exceptional', 'impressive',
            'delightful', 'pleased', 'satisfied', 'happy', 'joy', 'beautiful', 'quality', 'recommend',
            'worth', 'valuable', 'helpful', 'effective', 'efficient', 'comfortable', 'smooth',
            'reliable', 'trustworthy', 'innovative', 'creative', 'elegant', 'premium', 'superior',
            'flawless', 'magnificent', 'incredible', 'splendid', 'gorgeous', 'phenomenal'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'poor', 'hate', 'disappointing',
            'useless', 'waste', 'broken', 'defective', 'fake', 'fraud', 'scam', 'cheap',
            'unhappy', 'dissatisfied', 'angry', 'frustrated', 'annoying', 'difficult', 'complicated',
            'slow', 'expensive', 'overpriced', 'unreliable', 'failure', 'fail', 'problem',
            'issue', 'wrong', 'error', 'regret', 'unfortunately', 'never', 'avoid', 'pathetic',
            'disgusting', 'dreadful', 'inferior', 'mediocre', 'substandard', 'unacceptable'
        }
        
        self.negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 
                               'none', 'barely', 'hardly', 'scarcely', "don't", "doesn't", "didn't",
                               "won't", "wouldn't", "shouldn't", "can't", "cannot"}
        
        # Emotion Lexicon
        self.emotion_lexicon = {
            'joy': {'happy', 'joy', 'excited', 'delighted', 'pleased', 'cheerful', 'glad', 
                   'thrilled', 'ecstatic', 'elated', 'jubilant', 'overjoyed'},
            'anger': {'angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 
                     'outraged', 'enraged', 'livid', 'irate', 'infuriated'},
            'sadness': {'sad', 'unhappy', 'disappointed', 'depressed', 'miserable', 'upset', 
                       'sorry', 'heartbroken', 'melancholy', 'gloomy', 'sorrowful'},
            'fear': {'afraid', 'scared', 'worried', 'anxious', 'nervous', 'concerned', 
                    'frightened', 'terrified', 'alarmed', 'apprehensive', 'panicked'},
            'surprise': {'surprised', 'amazed', 'shocked', 'astonished', 'unexpected', 
                        'wow', 'astounded', 'stunned', 'startled'},
            'trust': {'trust', 'reliable', 'confident', 'believe', 'faithful', 'dependable',
                     'trustworthy', 'credible', 'honest', 'genuine'},
            'anticipation': {'expect', 'hope', 'anticipate', 'await', 'looking forward', 
                           'excited', 'eager', 'ready', 'prepared'}
        }
        
        # Initialize VADER if available
        if NLTK_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        
        self.results = []
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation for sentence structure
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        
        return text.strip()
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def lexicon_based_sentiment(self, text):
        """
        Custom lexicon-based sentiment analysis
        Returns: sentiment, confidence, scores
        """
        tokens = self.tokenize(text)
        
        pos_score = 0
        neg_score = 0
        
        # Check for negations
        negation_flag = False
        
        for i, token in enumerate(tokens):
            if token in self.negation_words:
                negation_flag = True
                continue
            
            if token in self.positive_words:
                if negation_flag:
                    neg_score += 1
                else:
                    pos_score += 1
            elif token in self.negative_words:
                if negation_flag:
                    pos_score += 1
                else:
                    neg_score += 1
            
            # Reset negation after 3 words
            if negation_flag and i > 0 and (i % 3 == 0):
                negation_flag = False
        
        # Calculate sentiment
        total_score = pos_score - neg_score
        
        if total_score > 0:
            sentiment = 'Positive'
            confidence = min(0.95, 0.5 + (total_score * 0.1))
        elif total_score < 0:
            sentiment = 'Negative'
            confidence = min(0.95, 0.5 + (abs(total_score) * 0.1))
        else:
            sentiment = 'Neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'pos_score': pos_score,
            'neg_score': neg_score,
            'total_score': total_score
        }
    
    def vader_sentiment(self, text):
        """VADER sentiment analysis"""
        if not NLTK_AVAILABLE:
            return None
        
        scores = self.vader.polarity_scores(text)
        
        # Determine sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(scores['compound']),
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg']
        }
    
    def textblob_sentiment(self, text):
        """TextBlob sentiment analysis"""
        if not TEXTBLOB_AVAILABLE:
            return None
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(polarity),
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def detect_emotions(self, text):
        """Detect emotions in text"""
        tokens = self.tokenize(text)
        emotions = {emotion: 0 for emotion in self.emotion_lexicon.keys()}
        
        for token in tokens:
            for emotion, words in self.emotion_lexicon.items():
                if token in words:
                    emotions[emotion] += 1
        
        # Get dominant emotion
        dominant_emotion = max(emotions, key=emotions.get) if max(emotions.values()) > 0 else 'neutral'
        
        return emotions, dominant_emotion
    
    def analyze_text(self, text, source='unknown'):
        """
        Comprehensive analysis of a single text
        """
        cleaned_text = self.preprocess_text(text)
        
        if not cleaned_text:
            return None
        
        # Lexicon-based analysis
        lexicon_result = self.lexicon_based_sentiment(cleaned_text)
        
        # VADER analysis
        vader_result = self.vader_sentiment(cleaned_text)
        
        # TextBlob analysis
        textblob_result = self.textblob_sentiment(cleaned_text)
        
        # Emotion detection
        emotions, dominant_emotion = self.detect_emotions(cleaned_text)
        
        # Ensemble sentiment (majority voting)
        sentiments = [lexicon_result['sentiment']]
        if vader_result:
            sentiments.append(vader_result['sentiment'])
        if textblob_result:
            sentiments.append(textblob_result['sentiment'])
        
        sentiment_counts = Counter(sentiments)
        ensemble_sentiment = sentiment_counts.most_common(1)[0][0]
        
        result = {
            'text': text[:200],  # Store first 200 chars
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'ensemble_sentiment': ensemble_sentiment,
            'lexicon': lexicon_result,
            'vader': vader_result,
            'textblob': textblob_result,
            'emotions': emotions,
            'dominant_emotion': dominant_emotion,
            'word_count': len(cleaned_text.split())
        }
        
        self.results.append(result)
        return result
    
    def analyze_batch(self, texts, sources=None):
        """Analyze multiple texts"""
        if sources is None:
            sources = ['unknown'] * len(texts)
        
        results = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"Processing {i}/{len(texts)}...")
            
            result = self.analyze_text(text, sources[i] if i < len(sources) else 'unknown')
            if result:
                results.append(result)
        
        return results
    
    def load_amazon_reviews(self, filepath, sample_size=1000):
        """Load Amazon reviews dataset"""
        try:
            # Try reading with different encodings
            try:
                df = pd.read_csv(filepath, encoding='utf-8', nrows=sample_size)
            except:
                df = pd.read_csv(filepath, encoding='latin-1', nrows=sample_size)
            
            # Common column names for Amazon reviews
            text_column = None
            for col in ['Text', 'text', 'review', 'Review', 'reviewText', 'review_text']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                print(f"Available columns: {df.columns.tolist()}")
                text_column = df.columns[0]
            
            texts = df[text_column].dropna().tolist()
            sources = ['Amazon'] * len(texts)
            
            print(f"Loaded {len(texts)} Amazon reviews")
            return self.analyze_batch(texts, sources)
        
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def load_twitter_data(self, filepath, sample_size=1000):
        """Load Twitter/social media dataset"""
        try:
            df = pd.read_csv(filepath, encoding='latin-1', nrows=sample_size)
            
            # Find text column
            text_column = None
            for col in ['text', 'tweet', 'Text', 'Tweet', 'content']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                text_column = df.columns[-1]
            
            texts = df[text_column].dropna().tolist()
            sources = ['Twitter'] * len(texts)
            
            print(f"Loaded {len(texts)} tweets")
            return self.analyze_batch(texts, sources)
        
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def get_statistics(self):
        """Calculate comprehensive statistics"""
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        
        stats = {
            'total_analyzed': len(self.results),
            'sentiment_distribution': df['ensemble_sentiment'].value_counts().to_dict(),
            'avg_confidence': df['lexicon'].apply(lambda x: x['confidence']).mean(),
            'emotion_distribution': {},
            'source_distribution': df['source'].value_counts().to_dict(),
            'dominant_emotions': df['dominant_emotion'].value_counts().to_dict()
        }
        
        # Aggregate emotions
        for emotion in self.emotion_lexicon.keys():
            stats['emotion_distribution'][emotion] = df['emotions'].apply(
                lambda x: x.get(emotion, 0)
            ).sum()
        
        return stats
    
    def visualize_results(self, save_path='sentiment_analysis.png'):
        """Create comprehensive visualizations"""
        if not self.results:
            print("No results to visualize")
            return
        
        stats = self.get_statistics()
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Sentiment Distribution (Pie Chart)
        ax1 = plt.subplot(2, 3, 1)
        sentiments = list(stats['sentiment_distribution'].keys())
        counts = list(stats['sentiment_distribution'].values())
        colors = ['#10b981' if s == 'Positive' else '#ef4444' if s == 'Negative' else '#6b7280' 
                  for s in sentiments]
        ax1.pie(counts, labels=sentiments, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # 2. Emotion Distribution (Bar Chart)
        ax2 = plt.subplot(2, 3, 2)
        emotions = list(stats['emotion_distribution'].keys())
        emotion_counts = list(stats['emotion_distribution'].values())
        bars = ax2.bar(emotions, emotion_counts, color='#8b5cf6')
        ax2.set_title('Emotion Detection', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Emotions')
        ax2.set_ylabel('Frequency')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. Source Distribution
        ax3 = plt.subplot(2, 3, 3)
        sources = list(stats['source_distribution'].keys())
        source_counts = list(stats['source_distribution'].values())
        ax3.bar(sources, source_counts, color='#3b82f6')
        ax3.set_title('Data Sources', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Source')
        ax3.set_ylabel('Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Sentiment over Time/Index (Line Chart)
        ax4 = plt.subplot(2, 3, 4)
        df = pd.DataFrame(self.results)
        sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        df['sentiment_score'] = df['ensemble_sentiment'].map(sentiment_mapping)
        
        # Calculate moving average
        window = max(len(df) // 20, 5)
        df['ma'] = df['sentiment_score'].rolling(window=window, min_periods=1).mean()
        
        ax4.plot(df.index, df['sentiment_score'], alpha=0.3, label='Raw', color='#94a3b8')
        ax4.plot(df.index, df['ma'], label=f'Moving Avg ({window})', color='#8b5cf6', linewidth=2)
        ax4.axhline(y=0, color='#6b7280', linestyle='--', alpha=0.5)
        ax4.set_title('Sentiment Trend', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Text Index')
        ax4.set_ylabel('Sentiment Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Dominant Emotions Distribution
        ax5 = plt.subplot(2, 3, 5)
        dom_emotions = list(stats['dominant_emotions'].keys())[:7]
        dom_counts = [stats['dominant_emotions'][e] for e in dom_emotions]
        ax5.barh(dom_emotions, dom_counts, color='#ec4899')
        ax5.set_title('Dominant Emotions', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Frequency')
        
        # 6. Confidence Distribution
        ax6 = plt.subplot(2, 3, 6)
        confidences = [r['lexicon']['confidence'] for r in self.results]
        ax6.hist(confidences, bins=30, color='#06b6d4', edgecolor='black', alpha=0.7)
        ax6.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Confidence Score')
        ax6.set_ylabel('Frequency')
        ax6.axvline(x=np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.2f}')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.show()
    
    def generate_insights(self):
        """Generate business insights from analysis"""
        if not self.results:
            return "No data analyzed yet."
        
        stats = self.get_statistics()
        
        total = stats['total_analyzed']
        sent_dist = stats['sentiment_distribution']
        
        positive_pct = (sent_dist.get('Positive', 0) / total) * 100
        negative_pct = (sent_dist.get('Negative', 0) / total) * 100
        neutral_pct = (sent_dist.get('Neutral', 0) / total) * 100
        
        # Generate insights
        insights = []
        
        insights.append("=" * 80)
        insights.append("SENTIMENT ANALYSIS INSIGHTS & RECOMMENDATIONS")
        insights.append("=" * 80)
        insights.append(f"\n📊 Dataset Overview:")
        insights.append(f"   • Total texts analyzed: {total:,}")
        insights.append(f"   • Average confidence: {stats['avg_confidence']:.2%}")
        
        insights.append(f"\n💭 Sentiment Breakdown:")
        insights.append(f"   • Positive: {positive_pct:.1f}% ({sent_dist.get('Positive', 0):,} texts)")
        insights.append(f"   • Negative: {negative_pct:.1f}% ({sent_dist.get('Negative', 0):,} texts)")
        insights.append(f"   • Neutral: {neutral_pct:.1f}% ({sent_dist.get('Neutral', 0):,} texts)")
        
        # Overall sentiment assessment
        insights.append(f"\n🎯 Overall Assessment:")
        if positive_pct > 60:
            insights.append("   ✅ EXCELLENT - Strong positive sentiment dominates")
            insights.append("   → Leverage positive feedback in marketing campaigns")
            insights.append("   → Identify and promote features customers love")
        elif positive_pct > 40:
            insights.append("   ✓ GOOD - Mostly positive with room for improvement")
            insights.append("   → Focus on converting neutral opinions to positive")
            insights.append("   → Address specific pain points from negative feedback")
        elif negative_pct > 40:
            insights.append("   ⚠  CONCERNING - High negative sentiment detected")
            insights.append("   → Immediate action required to address customer concerns")
            insights.append("   → Conduct root cause analysis of common complaints")
        else:
            insights.append("   ➖ MIXED - Balanced sentiment distribution")
            insights.append("   → Segment analysis needed to understand different user groups")
        
        # Emotion insights
        top_emotions = sorted(stats['emotion_distribution'].items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        
        insights.append(f"\n😊 Top Emotions Detected:")
        for i, (emotion, count) in enumerate(top_emotions, 1):
            insights.append(f"   {i}. {emotion.capitalize()}: {count} occurrences")
        
        # Actionable recommendations
        insights.append(f"\n💡 Actionable Recommendations:")
        
        if negative_pct > 20:
            insights.append("\n   🔴 For Product Development:")
            insights.append("      • Analyze negative reviews to identify common issues")
            insights.append("      • Prioritize bug fixes and feature improvements")
            insights.append("      • Set up automated alerts for sentiment drops")
        
        if positive_pct > 50:
            insights.append("\n   🟢 For Marketing:")
            insights.append("      • Extract testimonials from highly positive reviews")
            insights.append("      • Create case studies from satisfied customers")
            insights.append("      • Amplify positive sentiment on social media")
        
        insights.append("\n   🔵 For Customer Service:")
        insights.append("      • Reach out proactively to negative sentiment customers")
        insights.append("      • Train support team on common emotion patterns")
        insights.append("      • Implement sentiment-based ticket prioritization")
        
        # Trend recommendations
        insights.append("\n   📈 For Strategic Planning:")
        insights.append("      • Monitor sentiment trends weekly/monthly")
        insights.append("      • Compare sentiment across product lines/features")
        insights.append("      • Benchmark against competitor sentiment scores")
        
        insights.append("\n" + "=" * 80)
        
        return "\n".join(insights)
    
    def export_results(self, filename='sentiment_results.csv'):
        """Export results to CSV"""
        if not self.results:
            print("No results to export")
            return
        
        # Flatten results for CSV
        export_data = []
        for r in self.results:
            row = {
                'text': r['text'],
                'source': r['source'],
                'timestamp': r['timestamp'],
                'sentiment': r['ensemble_sentiment'],
                'dominant_emotion': r['dominant_emotion'],
                'confidence': r['lexicon']['confidence'],
                'positive_score': r['lexicon']['pos_score'],
                'negative_score': r['lexicon']['neg_score'],
                'word_count': r['word_count']
            }
            
            # Add emotion scores
            for emotion, score in r['emotions'].items():
                row[f'emotion_{emotion}'] = score
            
            # Add VADER scores if available
            if r['vader']:
                row['vader_compound'] = r['vader']['compound']
            
            # Add TextBlob scores if available
            if r['textblob']:
                row['textblob_polarity'] = r['textblob']['polarity']
            
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    def export_json(self, filename='sentiment_results.json'):
        """Export detailed results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Detailed results exported to {filename}")


# Main execution
def main():
    """Main execution function with examples"""
    
    print("=" * 80)
    print("ADVANCED SENTIMENT ANALYSIS SYSTEM")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Example 1: Analyze sample texts
    print("\n⿡  Analyzing Sample Reviews...")
    sample_reviews = [
        "This product is absolutely amazing! Best purchase I've ever made. Highly recommend!",
        "Terrible quality. Broke after one day. Complete waste of money. Very disappointed.",
        "It's okay. Nothing special but does the job. Average product for the price.",
        "Love it! Exceeded all my expectations. Great value and excellent customer service!",
        "Worst experience ever. Product arrived damaged and support was unhelpful.",
        "Not bad, but I expected better quality for the price. Somewhat disappointed.",
        "Incredible! This changed my life. I'm so happy with this purchase!",
        "Cheap plastic, poor design. Do not buy this. Save your money.",
        "Works fine. Does what it's supposed to do. Nothing more, nothing less.",
        "Outstanding product! Premium quality and fast shipping. Will buy again!"
    ]
    
    results = analyzer.analyze_batch(sample_reviews, ['Sample'] * len(sample_reviews))
    print(f"✓ Analyzed {len(results)} sample reviews")
    
    # Example 2: Analyze from CSV (if file exists)
    print("\n⿢  Loading from CSV (if available)...")
    print("   Looking for 'reviews.csv' or 'data.csv'...")
    
    for filename in ['reviews.csv', 'data.csv', 'amazon_reviews.csv']:
        try:
            results = analyzer.load_amazon_reviews(filename, sample_size=500)
            if results:
                print(f"✓ Successfully loaded data from {filename}")
                break
        except:
            continue
    else:
        print("   ℹ  No CSV file found. Using sample data only.")
    
    # Generate statistics
    print("\n⿣  Generating Statistics...")
    stats = analyzer.get_statistics()
    
    if stats:
        print(f"\n📊 Quick Stats:")
        print(f"   Total analyzed: {stats['total_analyzed']:,}")
        print(f"   Positive: {stats['sentiment_distribution'].get('Positive', 0)}")
        print(f"   Negative: {stats['sentiment_distribution'].get('Negative', 0)}")
        print(f"   Neutral: {stats['sentiment_distribution'].get('Neutral', 0)}")
    
    # Generate visualizations
    print("\n⿤  Creating Visualizations...")
    analyzer.visualize_results('sentiment_analysis_dashboard.png')
    
    # Generate insights
    print("\n⿥  Generating Business Insights...")
    insights = analyzer.generate_insights()
    print(insights)
    
    # Export results
    print("\n⿦  Exporting Results...")
    analyzer.export_results('sentiment_analysis_results.csv')
    analyzer.export_json('sentiment_analysis_detailed.json')
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  📊 sentiment_analysis_dashboard.png - Comprehensive visualizations")
    print("  📄 sentiment_analysis_results.csv - Tabular results")
    print("  📝 sentiment_analysis_detailed.json - Detailed JSON export")
    print("\n🔗 Recommended Datasets:")
    print("  1. https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews")
    print("  2. https://www.kaggle.com/datasets/bittlingmayer/amazonreviews")
    print("  3. https://www.kaggle.com/datasets/kazanova/sentiment140")
    print("=" * 80)


if _name_ == "_main_":
    main()


# Additional utility functions

def analyze_single_text(text):
    """Quick function to analyze a single piece of text"""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_text(text)
    
    print(f"\n📝 Text: {text[:100]}...")
    print(f"😊 Sentiment: {result['ensemble_sentiment']}")
    print(f"🎯 Confidence: {result['lexicon']['confidence']:.2%}")
    print(f"💭 Dominant Emotion: {result['dominant_emotion']}")
    print(f"📊 Emotion Breakdown: {result['emotions']}")
    
    return result


def compare_methods(text):
    """Compare different sentiment analysis methods on same text"""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_text(text)
    
    print(f"\n🔬 Comparing Methods for: '{text[:80]}...'\n")
    print(f"Lexicon-based: {result['lexicon']['sentiment']} (confidence: {result['lexicon']['confidence']:.2%})")
    
    if result['vader']:
        print(f"VADER: {result['vader']['sentiment']} (compound: {result['vader']['compound']:.3f})")
    
    if result['textblob']:
        print(f"TextBlob: {result['textblob']['sentiment']} (polarity: {result['textblob']['polarity']:.3f})")
    
    print(f"\n🎯 Ensemble Result: {result['ensemble_sentiment']}")
    
    return result


# Example usage scenarios
"""
# Scenario 1: Analyze your own data
analyzer = SentimentAnalyzer()
analyzer.load_amazon_reviews('your_reviews.csv', sample_size=1000)
analyzer.visualize_results()
print(analyzer.generate_insights())

# Scenario 2: Real-time analysis
text = "Your review or comment here"
result = analyze_single_text(text)

# Scenario 3: Batch processing
texts = ["review 1", "review 2", "review 3"]
analyzer = SentimentAnalyzer()
results = analyzer.analyze_batch(texts)
analyzer.export_results()

# Scenario 4: Compare different products
product_a_reviews = [...]  # Load reviews for product A
product_b_reviews = [...]  # Load reviews for product B
# Analyze both and compare sentiment distributions
"""
