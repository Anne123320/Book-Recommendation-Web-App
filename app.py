from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
from book_recommender import BookRecommendationSystem

app = Flask(__name__)

# Initialize the recommendation system
recommender = BookRecommendationSystem()

# Global variable to track if model is loaded
model_loaded = False

def verify_data_files():
    """Verify required data files exist"""
    required_files = ['Books.csv', 'Users.csv', 'Ratings.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required data files: {', '.join(missing_files)}. "
            f"Please ensure these files are in the same directory as app.py"
        )

def load_model_if_needed():
    """Load the model if not already loaded"""
    global model_loaded
    if not model_loaded:
        try:
            verify_data_files()
            # Try to load pre-trained model
            if not recommender.load_model('book_recommendation'):
                print("No saved model found. Loading data and training new model...")
                if not recommender.load_data('Books.csv', 'Users.csv', 'Ratings.csv'):
                    raise RuntimeError("Failed to load data")
                recommender.explore_data()
                recommender.create_popular_books()
                recommender.create_top_rated_books()
                recommender.create_author_books_list()
                recommender.prepare_collaborative_filtering()
                recommender.train_cosine_similarity_model()
                recommender.save_model('book_recommendation')
            model_loaded = True
            print("Model ready!")
        except Exception as e:
            print(f"Fatal error during model loading: {str(e)}")
            raise

@app.route('/')
def home():
    """Home page with popular books"""
    try:
        load_model_if_needed()
        
        # Get popular books for display
        popular_books = recommender.get_popular_books(6)
        if popular_books is None:
            raise ValueError("Could not load popular books")
            
        popular_books_list = []
        
        for _, book in popular_books.iterrows():
            popular_books_list.append({
                'title': book['Book-Title'],
                'author': book['Book-Author'],
                'image_url': book['Image-URL-M'],
                'avg_rating': round(book['avg_rating'], 1),
                'num_ratings': book['num_ratings']
            })
        
        return render_template('home.html', popular_books=popular_books_list)
        
    except Exception as e:
        return render_template('error.html', 
                            message="Failed to load popular books",
                            details=str(e)), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get book recommendations"""
    try:
        load_model_if_needed()
        
        book_title = request.form['book_title']
        n_recommendations = int(request.form.get('n_recommendations', 12))
        
        if not book_title or not book_title.strip():
            raise ValueError("Book title cannot be empty")
            
        # Get recommendations
        recommendations = recommender.get_recommendations(book_title.strip(), n_recommendations)
        
        # Get book details for the searched book
        searched_book = None
        book_data = recommender.books[
            recommender.books['Book-Title'].str.contains(book_title, case=False, na=False)
        ].drop_duplicates('Book-Title').head(1)
        
        if not book_data.empty:
            searched_book = {
                'title': book_data['Book-Title'].iloc[0],
                'author': book_data['Book-Author'].iloc[0],
                'image_url': book_data['Image-URL-M'].iloc[0]
            }
        
        return render_template('recommendations.html', 
                            recommendations=recommendations,
                            searched_book=searched_book,
                            search_query=book_title)
                            
    except Exception as e:
        return render_template('error.html',
                            message="Failed to get recommendations",
                            details=str(e)), 400
@app.route('/debug-top-rated')
def debug_top_rated():
    """Debug version to see what's failing"""
    try:
        # Test 1: Check if recommender exists
        if 'recommender' not in globals():
            return "ERROR: 'recommender' variable not found", 500
            
        # Test 2: Check if method exists
        if not hasattr(recommender, 'get_top_rated_books'):
            return "ERROR: 'get_top_rated_books' method not found on recommender", 500
            
        # Test 3: Try calling the method
        top_books = recommender.get_top_rated_books(30)
        return f"SUCCESS: Got data of type {type(top_books)}, shape/length: {getattr(top_books, 'shape', len(top_books) if top_books else 'None')}", 200
        
    except Exception as e:
        import traceback
        return f"ERROR: {str(e)}<br><br>Traceback:<br><pre>{traceback.format_exc()}</pre>", 500

@app.route('/top-rated')
def top_rated():
    """Top rated books page"""
    try:
        print("=== DEBUG: Starting top_rated route ===")
        
        print("Loading model...")
        load_model_if_needed()
        print("Model loaded successfully")
        
        print("Getting top rated books...")
        top_books = recommender.get_top_rated_books(30)
        print(f"Got response: {type(top_books)}")
        
        if top_books is None:
            print("ERROR: top_books is None")
            raise ValueError("Could not load top rated books")
        
        print(f"top_books shape: {top_books.shape if hasattr(top_books, 'shape') else 'No shape attribute'}")
        print(f"top_books columns: {list(top_books.columns) if hasattr(top_books, 'columns') else 'No columns attribute'}")
        
        top_books_list = []
        print("Starting to process books...")
        
        for index, book in top_books.iterrows():
            print(f"Processing book {index}: {book.get('Book-Title', 'Unknown Title')}")
            
            book_data = {
                'title': book.get('Book-Title', 'Unknown Title'),
                'author': book.get('Book-Author', 'Unknown Author'), 
                'image_url': book.get('Image-URL-M', 'https://via.placeholder.com/180x240?text=No+Image'),
                'avg_rating': round(float(book.get('avg_rating', 0)), 1),
                'num_ratings': int(book.get('num_ratings', 0))
            }
            top_books_list.append(book_data)
        
        print(f"Successfully processed {len(top_books_list)} books")
        
        print("Rendering template...")
        return render_template('top_rated.html', top_books=top_books_list)
        
    except Exception as e:
        print(f"=== ERROR in top_rated route ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        print("=== END ERROR ===")
        
        # Try to render error template, if that fails, return simple error
        try:
            return render_template('error.html', message="Failed to load top rated books", details=str(e)), 500
        except:
            return f"Error loading top rated books: {str(e)}", 500

@app.route('/api/search')
def api_search():
    """API endpoint for book search suggestions"""
    try:
        load_model_if_needed()
        
        query = request.args.get('q', '').lower().strip()
        if len(query) < 2:
            return jsonify([])
        
        # Find matching book titles with broader search
        books_df = recommender.books.drop_duplicates('Book-Title')
        matching_books = books_df[
            books_df['Book-Title'].str.contains(query, case=False, na=False)
        ]['Book-Title'].head(15).tolist()  # Increased from 10 to 15
        
        return jsonify(matching_books)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend/<book_title>')
def api_recommend(book_title):
    """API endpoint for getting recommendations"""
    try:
        load_model_if_needed()
        
        n_recommendations = request.args.get('n', 5, type=int)
        recommendations = recommender.get_recommendations(book_title.strip(), n_recommendations)
        
        return jsonify({
            'book_title': book_title,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("Starting Flask server...")
    print("Make sure you have the following files in your directory:")
    print("- Books.csv")
    print("- Users.csv") 
    print("- Ratings.csv")
    print("- book_recommender.py")
    print("- templates/home.html")
    print("- templates/recommendations.html")
    print("- templates/top_rated.html")
    print("- templates/error.html (for error handling)")
    
    try:
        verify_data_files()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")