# Dataset Information

## Harry Potter Fan Fiction Dataset

### Source
- **Original Source:** [Kaggle: Harry Potter Fanfiction Data](https://www.kaggle.com/datasets/nehatiwari03/harry-potter-fanfiction-data)
- **Data Origin:** Scraped from [fanfiction.net](https://www.fanfiction.net/book/Harry-Potter/)
- **Exercise Source:** CodeAcademy Logistic Regression Course

### Dataset Description
This dataset contains a cleaned subset of Harry Potter fan fiction stories with the following variables:

#### Quantitative Variables
- `words` - Number of words in the story
- `reviews` - Number of reviews the story received
- `favorites` - Number of readers who favorited the story
- `follows` - Number of readers who follow the story

#### Binary Categorical Variables (1 = True, 0 = False)
- `harry` - Harry is a character in the story
- `hermione` - Hermione is a character in the story
- `multiple` - The story has multiple chapters
- `english` - The story is in English
- `humor` - The story's genre is humor

### Getting the Dataset
1. **From CodeAcademy:** The `hp.csv` file should be available in your CodeAcademy exercise environment
2. **From Kaggle:** Download the full dataset from the Kaggle link above
3. **Place the file:** Save `hp.csv` in this `data/` directory

### Usage
The dataset is used for learning logistic regression with:
- Interaction terms (how variables work together)
- Polynomial terms (non-linear relationships)
- Model comparison (StatsModels vs scikit-learn)
- Feature engineering techniques

### Inspiration Questions
Based on the Kaggle dataset description, you can explore:
- Which is the most popular pairing?
- Which language has the most fan fiction written in it?
- What has been the general trend since the last movie or book came out?
- What factors make a fan fiction story popular?
