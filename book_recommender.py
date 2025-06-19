import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import warnings
import os
from scipy.sparse import csr_matrix
warnings.filterwarnings('ignore')

class BookRecommendationSystem:
    def __init__(self):
        self.books = None
        self.users = None
        self.ratings = None
        self.popular_df = None
        self.top_rated_df = None
        self.author_books_df = None
        self.pt = None
        self.similarity_scores = None
        self.model_svd = None
        
    def load_data(self, books_path='Books.csv', users_path='Users.csv', ratings_path='Ratings.csv'):
        """Load the datasets"""
        try:
            # Verify files exist before loading
            for path in [books_path, users_path, ratings_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Data file not found: {path}")
            
            self.books = pd.read_csv(books_path, encoding='latin1')
            self.users = pd.read_csv(users_path, encoding='latin1') 
            self.ratings = pd.read_csv(ratings_path, encoding='latin1')
            
            # Basic validation
            if len(self.books) == 0 or len(self.users) == 0 or len(self.ratings) == 0:
                raise ValueError("One or more data files are empty")
                
            print("Data loaded successfully!")
            print(f"Books shape: {self.books.shape}")
            print(f"Users shape: {self.users.shape}")
            print(f"Ratings shape: {self.ratings.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.books = None
            self.users = None
            self.ratings = None
            return False
    
    def explore_data(self):
        """Explore and clean the data"""
        if self.books is None or self.users is None or self.ratings is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n=== DATA EXPLORATION ===")
        
        # Display first few rows
        print("\nBooks head:")
        print(self.books.head())
        print("\nUsers head:")
        print(self.users.head())
        print("\nRatings head:")
        print(self.ratings.head())
        
        # Check for missing values
        print("\n=== MISSING VALUES ===")
        print(f"Books null values:\n{self.books.isnull().sum()}")
        print(f"\nUsers null values:\n{self.users.isnull().sum()}")
        print(f"\nRatings null values:\n{self.ratings.isnull().sum()}")
        
        # Check for duplicates
        print("\n=== DUPLICATES ===")
        print(f"Books duplicates: {self.books.duplicated().sum()}")
        print(f"Users duplicates: {self.users.duplicated().sum()}")
        print(f"Ratings duplicates: {self.ratings.duplicated().sum()}")
        
        # Clean data
        self.books.drop_duplicates(inplace=True)
        self.users.drop_duplicates(inplace=True)
        self.ratings.drop_duplicates(inplace=True)
        
        # Handle missing values in ratings
        self.ratings['Book-Rating'] = self.ratings['Book-Rating'].fillna(0)
        
        print("\nData cleaned!")
    
    def create_popular_books(self, min_ratings=25):
        """Create popular books dataframe"""
        if self.books is None or self.ratings is None:
            raise ValueError("Books and ratings data not loaded")
            
        print(f"\n=== CREATING POPULAR BOOKS (min {min_ratings} ratings) ===")
        
        try:
            # Merge ratings with books
            ratings_with_name = self.ratings.merge(self.books, on='ISBN')
            
            # Count number of ratings per book
            num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
            num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
            
            # Calculate average rating per book
            avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
            avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
            
            # Merge both dataframes
            popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
            
            # Filter books with minimum ratings and sort by average rating
            self.popular_df = popular_df[popular_df['num_ratings'] >= min_ratings].sort_values('avg_rating', ascending=False)
            
            # Add book details
            self.popular_df = self.popular_df.merge(
                self.books[['Book-Title', 'Book-Author', 'Image-URL-M']], 
                on='Book-Title'
            ).drop_duplicates('Book-Title')
            
            self.popular_df = self.popular_df[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']].head(50)
            
            print(f"Popular books created: {len(self.popular_df)} books")
            return self.popular_df
            
        except Exception as e:
            print(f"Error creating popular books: {e}")
            self.popular_df = None
            return None
    
    def create_top_rated_books(self, min_ratings=50):  # Lowered threshold
        """Create top-rated books dataframe"""
        if self.books is None or self.ratings is None:
            raise ValueError("Books and ratings data not loaded")
            
        print(f"\n=== CREATING TOP-RATED BOOKS (min {min_ratings} ratings) ===")
        
        try:
            ratings_with_name = self.ratings.merge(self.books, on='ISBN')
            
            # Count ratings and calculate average
            book_stats = ratings_with_name.groupby('Book-Title').agg({
                'Book-Rating': ['count', 'mean']
            }).reset_index()
            
            book_stats.columns = ['Book-Title', 'num_ratings', 'avg_rating']
            
            # Filter books with minimum ratings and decent average rating (>= 3.5)
            self.top_rated_df = book_stats[
                (book_stats['num_ratings'] >= min_ratings) & 
                (book_stats['avg_rating'] >= 3.5)  # Lowered from 4.0
            ].sort_values(['avg_rating', 'num_ratings'], ascending=[False, False])
            
            # Add book details
            self.top_rated_df = self.top_rated_df.merge(
                self.books[['Book-Title', 'Book-Author', 'Image-URL-M']],
                on='Book-Title'
            ).drop_duplicates('Book-Title')
            
            self.top_rated_df = self.top_rated_df[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']].head(100)  # Increased from 50
            
            print(f"Top-rated books created: {len(self.top_rated_df)} books")
            return self.top_rated_df
            
        except Exception as e:
            print(f"Error creating top-rated books: {e}")
            self.top_rated_df = None
            return None
    
    def create_author_books_list(self):
        """Create author-wise book list"""
        if self.books is None:
            raise ValueError("Books data not loaded")
            
        print("\n=== CREATING AUTHOR-WISE BOOK LIST ===")
        
        try:
            # Group books by author
            author_books = self.books.groupby('Book-Author')['Book-Title'].apply(list).reset_index()
            author_books['num_books'] = author_books['Book-Title'].apply(len)
            
            # Sort by number of books
            self.author_books_df = author_books.sort_values('num_books', ascending=False)
            
            print(f"Author-wise book list created: {len(self.author_books_df)} authors")
            return self.author_books_df
            
        except Exception as e:
            print(f"Error creating author books list: {e}")
            self.author_books_df = None
            return None
    
    def prepare_collaborative_filtering(self, min_user_ratings=50, min_book_ratings=20):
        """Prepare data for collaborative filtering with stricter filtering for smaller models"""
        if self.books is None or self.ratings is None:
            raise ValueError("Books and ratings data not loaded")
            
        print(f"\n=== PREPARING COLLABORATIVE FILTERING ===")
        print(f"Min user ratings: {min_user_ratings}, Min book ratings: {min_book_ratings}")
        
        try:
            # Merge ratings with book names
            ratings_with_name = self.ratings.merge(self.books, on='ISBN')
            
            # Filter active users (users who have rated many books) - increased threshold
            user_rating_counts = ratings_with_name.groupby('User-ID').count()['Book-Rating']
            active_users = user_rating_counts[user_rating_counts >= min_user_ratings].index
            
            # Filter the ratings to only include active users
            filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(active_users)]
            
            # Filter popular books (books that have been rated by many users) - increased threshold
            book_rating_counts = filtered_rating.groupby('Book-Title').count()['Book-Rating']
            famous_books = book_rating_counts[book_rating_counts >= min_book_ratings].index
            
            # Final filtered ratings
            final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
            
            print(f"Active users: {len(active_users)}")
            print(f"Famous books: {len(famous_books)}")
            print(f"Final ratings shape: {final_ratings.shape}")
            
            # Create pivot table
            self.pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
            self.pt.fillna(0, inplace=True)
            
            # Limit to top 1000 books to control model size
            if len(self.pt) > 1000:
                print(f"Limiting to top 1000 books (from {len(self.pt)})")
                # Select books with most ratings
                book_rating_sums = self.pt.sum(axis=1).sort_values(ascending=False)
                top_books = book_rating_sums.head(1000).index
                self.pt = self.pt.loc[top_books]
            
            print(f"Final pivot table shape: {self.pt.shape}")
            return self.pt
            
        except Exception as e:
            print(f"Error preparing collaborative filtering: {e}")
            self.pt = None
            return None
    
    def train_cosine_similarity_model(self):
        """Train cosine similarity based model with sparse storage optimization"""
        if self.pt is None:
            raise ValueError("Pivot table not prepared. Call prepare_collaborative_filtering() first")
            
        print("\n=== TRAINING COSINE SIMILARITY MODEL ===")
        
        try:
            # Convert to sparse matrix for memory efficiency
            sparse_pt = csr_matrix(self.pt.values)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(sparse_pt)
            
            # Only store similarities above a threshold to save space
            threshold = 0.1
            similarity_matrix[similarity_matrix < threshold] = 0
            
            # Convert to sparse format for storage
            self.similarity_scores = csr_matrix(similarity_matrix)
            
            print(f"Similarity matrix shape: {self.similarity_scores.shape}")
            print(f"Sparsity: {1 - (self.similarity_scores.nnz / (self.similarity_scores.shape[0] * self.similarity_scores.shape[1])):.3f}")
            print("Cosine similarity model trained successfully!")
            return True
            
        except Exception as e:
            print(f"Error training cosine similarity model: {e}")
            self.similarity_scores = None
            return False
    
    def train_matrix_factorization_model(self, n_components=50):
        """Train matrix factorization model using SVD"""
        if self.pt is None:
            raise ValueError("Pivot table not prepared. Call prepare_collaborative_filtering() first")
            
        print(f"\n=== TRAINING MATRIX FACTORIZATION MODEL (SVD) ===")
        
        try:
            # Apply SVD
            self.model_svd = TruncatedSVD(n_components=n_components, random_state=42)
            matrix_svd = self.model_svd.fit_transform(self.pt)
            
            # Calculate similarity using transformed matrix
            similarity_matrix = cosine_similarity(matrix_svd)
            
            # Apply threshold and convert to sparse
            threshold = 0.1
            similarity_matrix[similarity_matrix < threshold] = 0
            self.similarity_scores = csr_matrix(similarity_matrix)
            
            print(f"SVD model trained with {n_components} components")
            print(f"Explained variance ratio: {sum(self.model_svd.explained_variance_ratio_):.3f}")
            print(f"Similarity matrix sparsity: {1 - (self.similarity_scores.nnz / (self.similarity_scores.shape[0] * self.similarity_scores.shape[1])):.3f}")
            print("Matrix factorization model trained successfully!")
            return True
            
        except Exception as e:
            print(f"Error training matrix factorization model: {e}")
            self.model_svd = None
            self.similarity_scores = None
            return False
    
    def get_recommendations(self, book_title, n_recommendations=5, method='cosine'):
        """Get book recommendations with fuzzy matching"""
        if self.pt is None or self.similarity_scores is None:
            raise ValueError("Model not trained. Train the model first")
            
        if not book_title or not isinstance(book_title, str):
            raise ValueError("Invalid book title")
            
        try:
            # Normalize input
            book_title = book_title.strip()
            book_index = None
            matched_title = None
            
            # Try exact match first
            for i, title in enumerate(self.pt.index):
                if title.lower() == book_title.lower():
                    book_index = i
                    matched_title = title
                    break
            
            # If no exact match, try partial matching
            if book_index is None:
                for i, title in enumerate(self.pt.index):
                    if book_title.lower() in title.lower() or title.lower() in book_title.lower():
                        book_index = i
                        matched_title = title
                        break
            
            if book_index is None:
                print(f"Book '{book_title}' not found in the pivot table.")
                return []

            # Get similarity scores for the book (handle sparse matrix)
            if hasattr(self.similarity_scores, 'toarray'):
                similarity_row = self.similarity_scores[book_index].toarray().flatten()
            else:
                similarity_row = self.similarity_scores[book_index]
            
            similar_books = sorted(
                list(enumerate(similarity_row)),
                key=lambda x: x[1],
                reverse=True
            )[1:n_recommendations + 1]  # Exclude the book itself
            
            recommendations = []
            for i, score in similar_books:
                if score < 0.1:  # Lower threshold for more results
                    continue
                    
                book_data = self.books[self.books['Book-Title'] == self.pt.index[i]].drop_duplicates('Book-Title')
                if not book_data.empty:
                    book_info = {
                        'title': book_data['Book-Title'].iloc[0],
                        'author': book_data['Book-Author'].iloc[0],
                        'image_url': book_data['Image-URL-M'].iloc[0],
                        'similarity_score': round(score, 3)
                    }
                    recommendations.append(book_info)
            
            return recommendations
        
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def get_popular_books(self, n=10):
        """Get top n popular books"""
        if self.popular_df is None:
            if not self.create_popular_books():
                return None
        return self.popular_df.head(n)
    
    def get_top_rated_books(self, n=10):
        """Get top n rated books"""
        if self.top_rated_df is None:
            if not self.create_top_rated_books():
                return None
        return self.top_rated_df.head(n)
    
    def save_model(self, filepath_prefix='book_recommendation'):
        """Save the trained model and data with compression"""
        if (self.popular_df is None or self.top_rated_df is None or 
            self.author_books_df is None or self.pt is None or 
            self.books is None or self.similarity_scores is None):
            raise ValueError("Model not fully trained. Train all components first")
            
        print(f"\n=== SAVING MODEL ===")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath_prefix) or '.', exist_ok=True)
            
            # Use protocol 4 for better compression and compatibility
            protocol = pickle.HIGHEST_PROTOCOL
            
            # Save smaller dataframes
            with open(f'{filepath_prefix}_popular.pkl', 'wb') as f:
                pickle.dump(self.popular_df, f, protocol=protocol)
            
            with open(f'{filepath_prefix}_top_rated.pkl', 'wb') as f:
                pickle.dump(self.top_rated_df, f, protocol=protocol)
            
            with open(f'{filepath_prefix}_author_books.pkl', 'wb') as f:
                pickle.dump(self.author_books_df, f, protocol=protocol)
            
            # Save only essential book information to reduce size
            essential_books = self.books[['Book-Title', 'Book-Author', 'Image-URL-M', 'ISBN']].drop_duplicates()
            with open(f'{filepath_prefix}_books.pkl', 'wb') as f:
                pickle.dump(essential_books, f, protocol=protocol)
            
            # Save pivot table with compression
            with open(f'{filepath_prefix}_pivot_table.pkl', 'wb') as f:
                pickle.dump(self.pt, f, protocol=protocol)
            
            # Save sparse similarity matrix
            with open(f'{filepath_prefix}_similarity_scores.pkl', 'wb') as f:
                pickle.dump(self.similarity_scores, f, protocol=protocol)
            
            # Save SVD model if it exists
            if self.model_svd is not None:
                with open(f'{filepath_prefix}_svd_model.pkl', 'wb') as f:
                    pickle.dump(self.model_svd, f, protocol=protocol)
            
            # Print file sizes for reference
            for suffix in ['popular', 'top_rated', 'author_books', 'books', 'pivot_table', 'similarity_scores']:
                filepath = f'{filepath_prefix}_{suffix}.pkl'
                if os.path.exists(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"{suffix}: {size_mb:.2f} MB")
            
            print("Model saved successfully!")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath_prefix='book_recommendation'):
        """Load a pre-trained model"""
        print(f"\n=== LOADING MODEL ===")
        
        try:
            required_files = [
                f'{filepath_prefix}_popular.pkl',
                f'{filepath_prefix}_top_rated.pkl',
                f'{filepath_prefix}_author_books.pkl',
                f'{filepath_prefix}_pivot_table.pkl',
                f'{filepath_prefix}_books.pkl',
                f'{filepath_prefix}_similarity_scores.pkl'
            ]
            
            # Check all files exist
            for f in required_files:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Model file not found: {f}")
            
            with open(f'{filepath_prefix}_popular.pkl', 'rb') as f:
                self.popular_df = pickle.load(f)
            
            with open(f'{filepath_prefix}_top_rated.pkl', 'rb') as f:
                self.top_rated_df = pickle.load(f)
            
            with open(f'{filepath_prefix}_author_books.pkl', 'rb') as f:
                self.author_books_df = pickle.load(f)
            
            with open(f'{filepath_prefix}_pivot_table.pkl', 'rb') as f:
                self.pt = pickle.load(f)
            
            with open(f'{filepath_prefix}_books.pkl', 'rb') as f:
                self.books = pickle.load(f)
            
            with open(f'{filepath_prefix}_similarity_scores.pkl', 'rb') as f:
                self.similarity_scores = pickle.load(f)
            
            # Load SVD model if it exists
            svd_path = f'{filepath_prefix}_svd_model.pkl'
            if os.path.exists(svd_path):
                with open(svd_path, 'rb') as f:
                    self.model_svd = pickle.load(f)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Reset all model attributes if loading fails
            self.popular_df = None
            self.top_rated_df = None
            self.author_books_df = None
            self.pt = None
            self.books = None
            self.similarity_scores = None
            self.model_svd = None
            return False