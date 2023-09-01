import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer as tv
from sklearn.metrics.pairwise import cosine_similarity as cs


def MovieRecommender():
    # Reading the csv file
    movies_data = pd.read_csv(f'E:\\Python\\MovieRecommendation\\movies.csv')
    # print(movies_data.head())
    # print(movies_data.shape)

    # Selecting features to use for comparison
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

    # Replacing null values with null string
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # Combining all selected features
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + \
                        movies_data[
                            'cast'] + ' ' + movies_data['director']
    # print(combined_features)

    # Converting text data to feature vectors
    vectorizer = tv()
    feature_vectors = vectorizer.fit_transform(combined_features)
    # print(feature_vectors)

    # Getting similarity scores using cosine similarity
    similarity = cs(feature_vectors)
    # print(similarity)

    # Getting the movie name from the user
    movie_name = input("Enter movie name: ")

    # Getting a list of all movies' titles
    list_of_movies = movies_data['title'].tolist()
    # print(list_of_movies)

    # Finding the close match of the entered movie name
    find_close_match = difflib.get_close_matches(movie_name, list_of_movies)
    # print(find_close_match)

    # Getting the first movie from the close match list
    close_match = find_close_match[0]
    # print(close_match)

    # Finding the index of the movie with title
    index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    # print(index_of_movie)

    # Getting a list of similar movies
    similarity_score = list(enumerate(similarity[index_of_movie]))
    # print(similarity_score)

    # Sorting the list of similar movies in descending order
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    # print(sorted_similar_movies)

    # Printing the top 40 similar movies
    print('\nSuggested Movies: ')
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]

        if i < 41:
            print(i, '--> ', title_from_index)
            i += 1


if __name__ == "__main__":
    MovieRecommender()
