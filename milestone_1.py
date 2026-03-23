import pandas as pd
import re # text clening 
import string
import matplotlib.pyplot as plt

STOPWORDS={
    "is","the","and","to","a","an","of","in","for","on","with","as","by","at","from","that","this","it","be","are","was","from","or","but"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    words = text.split()  # split text into words
    words = [w for w in words if w not in STOPWORDS]  # remove stopwords
    return " ".join(words)  # join words back into a single string

def main():
    file_path_excel="ReviewSense_Customer_Feedback_5000.xlsx"
    file_path_csv="Milestone1_cleaned_feedback.csv"

    try:
        df = pd.read_excel(file_path_excel)
        print("Reading from Excel file")
    except FileNotFoundError:
        print("Excel file not found. Attempting to read from existing CSV.")
        try:
            df = pd.read_csv(file_path_csv)
            if "clean_feedback" in df.columns:
                print("CSV already has cleaned feedback. Skipping cleaning.")
            elif "feedback" in df.columns:
                print("Re-cleaning feedback from CSV.")
            else:
                raise ValueError("'feedback' column not found in CSV file.")
        except FileNotFoundError:
            raise ValueError("Neither Excel nor CSV file found. Please provide the input file.")

    if "feedback" not in df.columns:
        raise ValueError("'feedback' column not found in the file")

    # Ensure df is defined before using it
    df["clean_feedback"] = df["feedback"].apply(clean_text)
    df.to_csv("Milestone1_cleaned_feedback.csv", index=False)

    # Debugging: Check if 'clean_feedback' column is created correctly
    print("Clean Feedback Column:")
    print(df["clean_feedback"].head())

   
if __name__=="__main__":
    main()

